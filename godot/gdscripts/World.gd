extends Node2D

export(bool) var teleop_enabled = false

export(int) var followed_agent_id = null setget set_followed_agent_id

var agent_registry = {}
var pending_actions = []

# camera and related constants
onready var zoom = DEFAULT_ZOOM
onready var camera = null

export(float) var MAX_ZOOM_IN = 1
export(float) var MAX_ZOOM_OUT = 8
export(float) var DEFAULT_ZOOM = 2
export(float) var ZOOM_DELTA = 0.3

const ZOOM_IN_DIRECTION = -1
const ZOOM_OUT_DIRECTION = 1

# ZeroMQ Communication Settings
onready var agent_comm = $ZeroMqComm
onready var pub_context = null
onready var pub_options = {
	'agent':1234,
	'port':9001, 
	'protocol':'tcp'
}

onready var server_options = {
	'port':9002, 
	'protocol':'tcp'
}



# Called when the node enters the scene tree for the first time.
func _ready():
	create_world()
	
	# adds camera to agent
	camera = Camera2D.new()
	camera.current = true
	camera.zoom = Vector2(DEFAULT_ZOOM, DEFAULT_ZOOM)
	
	camera.smoothing_enabled = Globals.CAMERA_SMOOTHING_ENABLED
	if Globals.CAMERA_SMOOTHING_ENABLED:
		camera.smoothing_speed = Globals.CAMERA_SMOOTHING_SPEED
	
	var agents = $Agents.get_children()
	for agent in agents:
		add_agent_signal_handlers(agent)
		print("registering agent with id ", agent.id)
		agent_registry[agent.id] = agent
	
	set_followed_agent_id(agents[0].id)
	
	init_agent_comm()
	
	

func init_agent_comm():
	pub_context = agent_comm.connect(pub_options)
	agent_comm.start_listener(server_options)
	
	
func _process(delta):
	
	# check for actions from human-controlled agents
	if teleop_enabled:
		process_teleop_action()

	var n_food_to_spawn = Globals.N_RANDOM_FOOD - $Food.get_child_count()
	if (n_food_to_spawn > 0):
		spawn(n_food_to_spawn, "res://scenes/Food.tscn", $Food)
	
	# check

func has_collision(obj):
	var space_rid = get_world_2d().space
	var space_state = Physics2DServer.space_get_direct_state(space_rid)

	var shape = obj.get_node("CollisionShape2D").shape

	# translates shape's location to the object's global position
	var transform = Transform2D()
	transform.origin = obj.global_position
		
	var query = Physics2DShapeQueryParameters.new()
	query.set_shape(shape)
	query.set_transform(transform)
	query.set_exclude([obj])
	query.collision_layer = Globals.OBSTACLES_LAYER | Globals.AGENTS_LAYER | Globals.EDIBLES_LAYER

	var results = space_state.intersect_shape(query, 1)
	
#	for result in results:
#		print(result)

	return len(results) > 0

func create_objects(scene, locations, parent):
	for loc in locations:
		var obj = load(scene).instance()
		obj.position = loc
		obj.visible = false

		# the node must be added to the scene tree before collision detection or
		# the following error will occur: 
		# ---> ERROR: get_global_transform: Condition "!is_inside_tree()" is true. Returned: get_transform()
		parent.add_child(obj)
			
		# check for collision with existing object
		if has_collision(obj):
			obj.free()
		else:
			obj.visible = true	

func spawn(n, scene, parent):
	
	for i in range(n):
		create_objects(scene, 
			[Vector2(rand_range(Globals.WORLD_HORIZ_EXTENT[0], Globals.WORLD_HORIZ_EXTENT[1]), 
					 rand_range(Globals.WORLD_VERT_EXTENT[0], Globals.WORLD_VERT_EXTENT[1]))], 
			parent)	
	
func create_random_objects():
	
	for i in range(Globals.N_RANDOM_FOOD):
		create_objects("res://scenes/Food.tscn", 
			[Vector2(rand_range(Globals.WORLD_HORIZ_EXTENT[0], Globals.WORLD_HORIZ_EXTENT[1]), 
					 rand_range(Globals.WORLD_VERT_EXTENT[0], Globals.WORLD_VERT_EXTENT[1]))], 
			$Food)

#	for i in range(Globals.N_RANDOM_ROCKS):
#		create_objects("res://objects/simple/rock-obstacle.tscn", 
#			[Vector2(rand_range(Globals.WORLD_HORIZ_EXTENT[0], Globals.WORLD_HORIZ_EXTENT[1]),
#					 rand_range(Globals.WORLD_VERT_EXTENT[0], Globals.WORLD_VERT_EXTENT[1]))], 
#			$Rocks)

	# this should never print anything. if it does, then there may be a memory leak
	print_stray_nodes()	
							
func create_world():
		# TODO: Load objects from a saved world file (e.g., a JSON file)
#	load_objects()
	
	if Globals.RANDOMIZED:
		create_random_objects()	
		
	# installs signal handlers
#	for node in $Food.get_children():
#		add_food_signal_handlers(node)
		
	for node in $Agents.get_children():
		add_agent_signal_handlers(node)	

func set_followed_agent_id(id):
	if followed_agent_id != null:
		agent_registry[followed_agent_id].remove_child(camera)

	if agent_registry.has(id):
		agent_registry[id].add_child(camera)	
		followed_agent_id = id
	
func zoom_in():
	print('zooming in')
	if camera and zoom - ZOOM_DELTA >= MAX_ZOOM_IN:
		zoom -= ZOOM_DELTA
		update_camera_zoom(ZOOM_IN_DIRECTION)
	
func zoom_out():
	print('zooming out')
	if camera and zoom + ZOOM_DELTA <= MAX_ZOOM_OUT:
		update_camera_zoom(ZOOM_OUT_DIRECTION)
		
func update_camera_zoom(direction):
	zoom += direction * ZOOM_DELTA
	camera.zoom = Vector2(zoom, zoom)
	
func _input(event):	
	if event.is_action_pressed("ui_zoom_in"):
		zoom_in()
	elif event.is_action_pressed("ui_zoom_out"):
		zoom_out()

func add_agent_signal_handlers(agent):
	agent.connect(
		"agent_consumed_edible", 
		self, 
		"_on_agent_consumed_edible")		
	
func process_teleop_action():
	var actions = {}
	
	actions[Globals.AGENT_ACTIONS.FORWARD] = true if Input.is_action_pressed("ui_up") else false;
	actions[Globals.AGENT_ACTIONS.BACKWARD] = true if Input.is_action_pressed("ui_down") else false;
	actions[Globals.AGENT_ACTIONS.TURN_LEFT] = true if Input.is_action_pressed("ui_left") else false;
	actions[Globals.AGENT_ACTIONS.TURN_RIGHT] = true if Input.is_action_pressed("ui_right") else false;

	var agent = agent_registry[followed_agent_id]
	if agent:
		agent.add_action(actions)

func _on_agent_consumed_edible(agent, edible):
	print("agent %s consumed edible %s" % [agent.id, edible])
	
	edible.call_deferred("queue_free")
		
	# satiety is always increased after consuming edibles
	agent.stats.satiety += Globals.SATIETY_PER_UNIT_FOOD
	
	agent.print_stats()
		
func _on_remote_action_received(action_details):
	var id = action_details['agent_id']
	var actions = action_details['actions']
	
	# ignore incoming remote actions for human controlled agent
	if (teleop_enabled and id == followed_agent_id):
		return
		
	var agent = agent_registry[id]
	if agent:
		agent.add_action(actions)
