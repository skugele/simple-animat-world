extends KinematicBody2D

# a unique identifier for the agent
onready var id = null

# a list of actions pending execution
onready var pending_actions = []

###############
# sensor vars #
###############
onready var active_scents = {} # dictionary of scent emitters ids -> active scent areas
onready var active_scents_combined = Globals.NULL_SMELL

onready var ignored_scents = [] # ignore own smells
onready var combined_scent_sig = null

onready var active_tactile_events = 0

###########
# signals #
###########
signal agent_consumed_edible(agent, edible)


# current agent state vars
onready var velocity = Vector2.ZERO setget set_velocity
onready var stats = $AgentStats

###############################
# built-in function overrides #
###############################

# Called when the node enters the scene tree for the first time.
func _ready():
	id = 1 # Globals.generate_unique_id()

func _process(delta):

	# (1) execute next pending action(s) -- parallel action execution is possible
	var actions = pending_actions.pop_front()
	if actions:
		execute(actions, delta)
		
	# (2) update state variables
	active_scents_combined = get_combined_scent(active_scents)
		
		
	
func print_stats():	
	print('agent %s\'s health: %s' % [id, stats.health])
	print('agent %s\'s satiety: %s' % [id, stats.satiety])
		
func add_action(action):
	if len(pending_actions) > Globals.MAX_PENDING_ACTIONS:
		pending_actions.pop_front()
		print('Max queue depth reached. Dropping oldest pending action.')
		
	pending_actions.push_back(action)

func execute(actions, delta):
#	print('executing actions: ', actions)
	var turn = 0
	var direction = Vector2.ZERO
	
	# FIXME: this is a hack to allow a different forward and reverse max speed
	var max_speed = 0

	# linear motion	
	
	if actions[Globals.AGENT_ACTIONS.BACKWARD] and actions[Globals.AGENT_ACTIONS.FORWARD]:
		# both forward and backward actions together result in a net-zero linear displacement	
		pass		
	
	elif actions[Globals.AGENT_ACTIONS.FORWARD]:
		direction = Vector2(0, -1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_FORWARD		
	
	elif actions[Globals.AGENT_ACTIONS.BACKWARD]:		
		direction = Vector2(0, 1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_BACKWARD	
		
#
#	# angular movement
	if actions[Globals.AGENT_ACTIONS.TURN_RIGHT] and actions[Globals.AGENT_ACTIONS.TURN_LEFT]:
		# both right and left turn actions together result in a net-zero turn	
		pass		
		
	elif actions[Globals.AGENT_ACTIONS.TURN_RIGHT]:
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

func add_scent(scent):	
#	print('agent %s smells scent %s with signature %s' % [self, scent, scent.signature])
	
	if active_scents.has(scent.smell_emitter_id):
		active_scents[scent.smell_emitter_id].push_back(scent)
	else:
		active_scents[scent.smell_emitter_id] = [scent]

func remove_scent(scent):
#	print('agent %s lost scent %s with signature %s' % [self, scent, scent.signature])
	
	if len(active_scents[scent.smell_emitter_id]) <= 1:
		active_scents.erase(scent.smell_emitter_id)
	else:
		var removed_scent = active_scents[scent.smell_emitter_id].pop_back()
	
func distance_from_scent(scent):
	if scent == null:
		return null
		
	var distance = Globals.SMELL_DETECTABLE_RADIUS
	var detector_pos = $SmellDetector.global_position
		
	if scent.is_inside_tree():
		var scent_pos = scent.global_position		
		distance = min(distance, detector_pos.distance_to(scent_pos))
	
	return distance
	
func get_combined_scent(active_scents):
	var combined_scent_sig = Globals.NULL_SMELL

	for id in active_scents.keys():
		var scent = active_scents[id][-1]
		
		var distance = distance_from_scent(scent)
		if distance == null:
			continue

		# maps distance onto the range [0, 1]
		var d_effective = distance / Globals.SMELL_DETECTABLE_RADIUS
		var scaling_factor = 1.0 / (pow(Globals.SMELL_DISTANCE_MULTIPLIER * d_effective, 
										Globals.SMELL_DISTANCE_EXPONENT) + 1.0)

		var scaled_scent = Globals.scale(scent.signature, scaling_factor)

		combined_scent_sig = Globals.add_vectors(combined_scent_sig, scaled_scent)
		
	return combined_scent_sig

###################
# Signal Handlers #
###################
func _on_smell_detected(scent):
	add_scent(scent)

func _on_smell_lost(scent):
	remove_scent(scent)

func _on_edible_consumed(edible):
	emit_signal("agent_consumed_edible", self, edible)

func _on_tactile_event(body):
	if body != self:
		active_tactile_events += 1

func _on_tactile_event_ends(body):
	if body != self:
		active_tactile_events -= 1


func _on_death():
	print("Agent {} is dead!".format(id))
