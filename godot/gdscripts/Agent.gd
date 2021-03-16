extends KinematicBody2D

################
# onready vars #
################
onready var id = null

onready var pending_actions = []

###########
# signals #
###########
signal agent_consumed_edible(agent, edible)


# current agent state vars
onready var velocity = Vector2.ZERO setget set_velocity
onready var stats = $AgentStats

# Called when the node enters the scene tree for the first time.
func _ready():
	id = Globals.generate_unique_id()

func _process(delta):
	var actions = pending_actions.pop_front()
	if actions:
		execute(actions, delta)
	
func print_stats():	
	print('agent %s\'s health: %s' % [id, stats.health])
	print('agent %s\'s satiety: %s' % [id, stats.satiety])
		
func add_action(action):
	pending_actions.push_back(action)

func execute(actions, delta):
#	print('executing actions: ', actions)
	var turn = 0
	var direction = Vector2.ZERO
	
	# FIXME: this is a hack to allow a different forward and reverse max speed
	var max_speed = 0

	# forward/backward motion
	if actions[Globals.AGENT_ACTIONS.FORWARD]:
		direction = Vector2(0, -1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_FORWARD	
	elif actions[Globals.AGENT_ACTIONS.BACKWARD]:
		direction = Vector2(0, 1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_BACKWARD	
#
#	# body rotation
	if actions[Globals.AGENT_ACTIONS.TURN_RIGHT]:
		turn += 1.0
	elif actions[Globals.AGENT_ACTIONS.TURN_LEFT]:
		turn -= 1.0

	# execute forward/backward motion
	if direction != Vector2.ZERO:
		update_velocity(direction, delta, max_speed)
	else:
		# apply friction
		velocity = velocity.move_toward(Vector2.ZERO, Globals.AGENT_WALKING_FRICTION * delta)

	# execute body rotation
	if turn != 0:
		update_rotation(turn, delta)

	velocity = move_and_slide(velocity)
	set_velocity(velocity)

func update_velocity(direction, delta, max_speed):
	velocity = velocity.move_toward(direction * max_speed, Globals.AGENT_WALKING_ACCELERATION * delta)
	
func update_rotation(turn, delta):
	rotation += turn * Globals.AGENT_MAX_ROTATION_RATE * delta
	
func set_rotation(degrees):
	rotation = degrees
	
func set_velocity(value):
	velocity = value

func _on_edible_consumed(edible):
	emit_signal("agent_consumed_edible", self, edible)
