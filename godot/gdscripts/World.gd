extends Node2D

export(bool) var teleop_enabled = false

var followed_agent = null

var agent_registry = {}
var pending_actions = []

# camera and related constants
onready var zoom = DEFAULT_ZOOM
onready var camera = null

export(float) var MAX_ZOOM_IN = 1
export(float) var MAX_ZOOM_OUT = 16
export(float) var DEFAULT_ZOOM = 2
export(float) var ZOOM_DELTA = 0.4

const ZOOM_IN_DIRECTION = -1
const ZOOM_OUT_DIRECTION = 1

# ZeroMQ Communication Settings
onready var agent_comm = $ZeroMqComm
onready var pub_context = null
onready var pub_options = {
	'port':9001, 
	'protocol':'tcp'
}

# TODO: This doesn't do anything yet. Need to modify action listener to accept a dictionary of arguments
onready var server_options = {
	'port':9002, 
	'protocol':'tcp'
}

onready var initialized = false

# Called when the node enters the scene tree for the first time.
func _ready():
	init_camera()
	init_agent_comm()
	
	create_world()
	
	initialized = true

func init_camera():
	camera = Camera2D.new()
	camera.current = true
	camera.zoom = Vector2(DEFAULT_ZOOM, DEFAULT_ZOOM)
	
	camera.smoothing_enabled = Globals.CAMERA_SMOOTHING_ENABLED
	if Globals.CAMERA_SMOOTHING_ENABLED:
		camera.smoothing_speed = Globals.CAMERA_SMOOTHING_SPEED	
	
	$Observer.set_camera(camera)
	
func init_agent_comm():
	pub_context = agent_comm.connect(pub_options)
	agent_comm.start_listener(server_options)	
	

func _process(delta):
	# check for actions from human-controlled agents
	if teleop_enabled and followed_agent:
		process_teleop_action()
	
func _physics_process(delta):
	
	var n_food_to_spawn = Globals.N_RANDOM_FOOD - $Food.get_child_count()
	if (n_food_to_spawn > 0):
		spawn(n_food_to_spawn, "res://scenes/Food.tscn", $Food)
	
	for agent in agent_registry.values():
		publish_sensors_state(agent)


func get_sensors_topic(agent):
	return Globals.SENSORS_STATE_TOPIC_FORMAT.format({'agent_id':agent.id})
	
func get_sensors_message(agent):
	var msg = {}
	
	# add smell sensor
	msg[Globals.OLFACTORY_SENSOR_ID] = agent.active_scents_combined
	msg[Globals.TACTILE_SENSOR_ID] = 1 if agent.active_tactile_events else 0
#	msg[Globals.SOMATO_SENSOR_ID] = agent.stats.as_list()
	msg[Globals.SOMATO_SENSOR_ID] = agent.stats.satiety
	
	return msg
	
func publish_sensors_state(agent):
	var topic = get_sensors_topic(agent)
	var msg = get_sensors_message(agent)
	
#	print('publishing to topic ', topic)
#	print('message: ', msg)
	agent_comm.send(msg, pub_context, topic)

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
	
	for obj in range(n):
		create_objects(scene, 
			[Vector2(rand_range(Globals.WORLD_HORIZ_EXTENT[0], Globals.WORLD_HORIZ_EXTENT[1]), 
					 rand_range(Globals.WORLD_VERT_EXTENT[0], Globals.WORLD_VERT_EXTENT[1]))], 
			parent)	
			
func create_random_objects():
	
	var n_agents_to_spawn = Globals.N_AGENTS - $Agents.get_child_count()
	if (n_agents_to_spawn > 0):
		spawn(n_agents_to_spawn, "res://scenes/Agent.tscn", $Agents)
		
	var n_food_to_spawn = Globals.N_RANDOM_FOOD - $Food.get_child_count()
	if (n_food_to_spawn > 0):
		spawn(n_food_to_spawn, "res://scenes/Food.tscn", $Food)
		
	# this should never print anything. if it does, then there may be a memory leak
	print_stray_nodes()	
							
func create_world():
	# TODO: Load objects from a saved world file (e.g., a JSON file)
#	load_objects()
	
	if Globals.RANDOMIZED:
		create_random_objects()	
		
	init_objects()	

func init_objects():
	# initialize agents
	var agents = $Agents.get_children()
	for agent in agents:
		add_agent_signal_handlers(agent)
		agent_registry[agent.id] = agent
	
func follow_next_agent():
	if not followed_agent:
		follow_agent(0)
	else:
		follow_agent((followed_agent.id + 1) % len(agent_registry))

func follow_prev_agent():
	if not followed_agent:
		follow_agent(0)
	else:
		follow_agent((followed_agent.id - 1) % len(agent_registry))
		
func follow_observer():
	# remove camera from previously followed agent (if any)
	if followed_agent:
		followed_agent.unset_camera(camera)
	
	$Observer.set_camera(camera)	
	
func follow_agent(id):
	if followed_agent and id == followed_agent.id:
		return
		
	# remove camera from previously followed agent (if any)
	if followed_agent:
		followed_agent.unset_camera(camera)

	# add camera to new followed agent
	if agent_registry.has(id):
		followed_agent = agent_registry[id]

		$Observer.unset_camera(camera)
		followed_agent.set_camera(camera)
	else:
		$Observer.set_camera(camera)			

func zoom_in():
	if camera and zoom - ZOOM_DELTA >= MAX_ZOOM_IN:
		zoom -= ZOOM_DELTA
		update_camera_zoom(ZOOM_IN_DIRECTION)
	
func zoom_out():
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
	elif event.is_action_pressed("next_agent"):
		follow_next_agent()
	elif event.is_action_pressed("prev_agent"):
		follow_prev_agent()
	elif event.is_action_pressed("follow_observer"):
		follow_observer()
		
	# Process observer actions
	else:
		
		if $Observer.has_camera:
			if Input.is_action_pressed("ui_up"):
				$Observer.move_up()
			
			if Input.is_action_pressed("ui_down"):
				$Observer.move_down()
			
			if Input.is_action_pressed("ui_left"):
				$Observer.move_left()
			
			if Input.is_action_pressed("ui_right"):
				$Observer.move_right()

func add_agent_signal_handlers(agent):
	agent.connect(
		"agent_consumed_edible", 
		self, 
		"_on_agent_consumed_edible")		
	
func process_teleop_action():	
	var actions = 0
	
	if Input.is_action_pressed("ui_up"):
		actions |= Globals.AGENT_ACTIONS.FORWARD
		
	if Input.is_action_pressed("ui_down"):
		actions |= Globals.AGENT_ACTIONS.BACKWARD	
		
	if Input.is_action_pressed("ui_left"):
		actions |= Globals.AGENT_ACTIONS.TURN_LEFT
		
	if Input.is_action_pressed("ui_right"):
		actions |= Globals.AGENT_ACTIONS.TURN_RIGHT	
	
	if followed_agent and actions > 0:
		followed_agent.add_action(actions)
				
func _on_agent_consumed_edible(agent, edible):
	print("agent %s consumed edible %s" % [agent.id, edible])
	
	edible.call_deferred("queue_free")
		
	# satiety is always increased after consuming edibles
	agent.stats.satiety += Globals.SATIETY_PER_UNIT_FOOD
		
func _on_remote_action_received(action_details):
#	print('received message: ', action_details)
	
	if not (action_details.has('id') and action_details.has('action')):
		print('malformed message: %s expecting \'id\' and \'actions\'' % action_details)
		return
		
	var id = action_details['id']	
	var actions = action_details['action']
	
	# ignore incoming remote actions for human controlled agent
	if (teleop_enabled and id == followed_agent.id):
		return
		
	var agent = agent_registry[id]
	if agent:
		agent.add_action(actions)
