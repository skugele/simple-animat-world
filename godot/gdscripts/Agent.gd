extends KinematicBody2D

# a unique identifier for the agent
export onready var id = null

# a list of actions pending execution
onready var pending_actions = []

###############
# sensor vars #
###############
onready var antenna_left = $AntennaLeft
onready var antenna_right = $AntennaRight

onready var active_scents_left = {} # dictionary of scent emitters ids -> active scent areas
onready var active_scents_right = {} # dictionary of scent emitters ids -> active scent areas

onready var active_scents_left_combined = Globals.NULL_SMELL
onready var active_scents_right_combined = Globals.NULL_SMELL
onready var active_scents_combined = [0.0, 0.0]

onready var ignored_scents = [] # ignore own smells

onready var combined_scent_sig_left = null
onready var combined_scent_sig_right = null

onready var active_tactile_events = 0

onready var has_camera = false

onready var last_action_seqno = -1

onready var pending_actions_mutex = Mutex.new()
onready var pending_actions_semaphore = Semaphore.new()

###########
# signals #
###########
signal agent_consumed_edible(agent, edible)

# current agent state vars
onready var velocity = Vector2.ZERO setget set_velocity
onready var stats = $AgentStats

onready var action_thread = null

onready var agent_alive = true

###############################
# built-in function overrides #
###############################

# Called when the node enters the scene tree for the first time.
func _ready():
	pass
#	id = Globals.generate_agent_id()
	
#	action_thread = Thread.new()
#	action_thread.start(self, "_process_actions")

func reset():
	stats.reset()
	last_action_seqno = -1

func enable():
	self.visible = true
	$CollisionShape2D.disabled = false
	$Mouth/Area2D/CollisionShape2D.disabled = false
	$AntennaLeft/Area2D/CollisionShape2D.disabled = false
	$AntennaRight/Area2D/CollisionShape2D.disabled = false
	$TactileSensor/CollisionShape2D.disabled = false
	
func disable():
	self.visible = false
	$CollisionShape2D.disabled = true
	$Mouth/Area2D/CollisionShape2D.disabled = true
	$AntennaLeft/Area2D/CollisionShape2D.disabled = true
	$AntennaRight/Area2D/CollisionShape2D.disabled = true
	$TactileSensor/CollisionShape2D.disabled = true
	
func _exit_tree():
	pass
	# safely clean-up action thread
#	agent_alive = false
#	pending_actions_semaphore.post()
#	action_thread.wait_to_finish()
	
func _process(delta):
	var pending_action = pending_actions.pop_back()	
	if pending_action:
		execute(pending_action['action'], delta)
		velocity = move_and_slide(velocity)		
		last_action_seqno = pending_action['seqno']
	else:
		# process friction		
		velocity = velocity.move_toward(Vector2.ZERO, Globals.AGENT_WALKING_FRICTION * delta)		
	
	if velocity != Vector2.ZERO:
		velocity = move_and_slide(velocity)
		
	# (2) update state variables
	active_scents_left_combined = get_combined_scent($AntennaLeft/Area2D, active_scents_left)
	active_scents_right_combined = get_combined_scent($AntennaRight/Area2D, active_scents_right)

	active_scents_combined = []	
	active_scents_combined += active_scents_left_combined
	active_scents_combined += active_scents_right_combined
	
func print_stats():	
	print('agent %s\'s health: %s' % [id, stats.health])
	print('agent %s\'s satiety: %s' % [id, stats.satiety])
		
func add_action(seqno, action):
	if seqno != -1 and seqno <= last_action_seqno:
		return
	
#	pending_actions_mutex.lock()
	if len(pending_actions) > Globals.MAX_PENDING_ACTIONS:
		var dropped_actions = pending_actions.pop_back()
		print('Max queue depth reached. Dropping oldest pending action with value %s.' % dropped_actions)
		
	pending_actions.push_front({'seqno': seqno, 'action': action})
#	pending_actions_mutex.unlock()
#	pending_actions_semaphore.post()

# actions are encodings of parallel actions as integers in the range [0, 2^(N_ACTIONS) - 1). bitwise operations
# are used to determine the actions to execute. For example, the integer 13d = 1101b indicates that actions 1, 3, and 4
# are to be executed in parallel.
func execute(action, delta):	
	
	if action == Globals.NOOP_ACTION:
		return
		
#	print('executing actions: ', actions)
	var turn = 0
	var direction = Vector2.ZERO
	
	# FIXME: this is a hack to allow a different forward and reverse max speed
	var max_speed = 0

	# linear motion	
	if action & Globals.AGENT_ACTIONS.BACKWARD and action & Globals.AGENT_ACTIONS.FORWARD:
		# both forward and backward actions together result in a net-zero linear displacement	
		pass		
	
	elif action & Globals.AGENT_ACTIONS.FORWARD:
		direction = Vector2(0, -1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_FORWARD
	
	elif action & Globals.AGENT_ACTIONS.BACKWARD:		
		direction = Vector2(0, 1).rotated(rotation)
		max_speed = Globals.AGENT_MAX_SPEED_BACKWARD
		
#
#	# angular movement
	if action & Globals.AGENT_ACTIONS.TURN_RIGHT and action & Globals.AGENT_ACTIONS.TURN_LEFT:
		# both right and left turn actions together result in a net-zero turn	
		pass		
		
	elif action & Globals.AGENT_ACTIONS.TURN_RIGHT:
		turn += 1.0
		
	elif action & Globals.AGENT_ACTIONS.TURN_LEFT:
		turn -= 1.0

	# execute forward/backward motion
	if direction != Vector2.ZERO:
		update_velocity(direction, delta, max_speed)
	else:
		update_velocity(Vector2.ZERO, delta, Globals.AGENT_WALKING_FRICTION)		

	# execute body rotation
	if turn != 0:
		update_rotation(turn, delta)
		
func update_velocity(direction, delta, max_speed):
	velocity = velocity.move_toward(direction * max_speed, Globals.AGENT_WALKING_ACCELERATION * delta)
	
func update_rotation(turn, delta):
	rotation += turn * Globals.AGENT_MAX_ROTATION_RATE * delta
	
func set_rotation(degrees):
	rotation = degrees
	
func set_velocity(value):
	velocity = value

func set_camera(camera):
	if not has_camera:
		add_child(camera)
		has_camera = true
		
func unset_camera(camera):
	if has_camera:
		remove_child(camera)
		has_camera = false
		
func add_scent(origin, scent):	
#	print('agent %s smells scent %s with signature %s' % [self, scent, scent.signature])
	
	if origin.id == antenna_left.id:
		if active_scents_left.has(scent.smell_emitter_id):
			active_scents_left[scent.smell_emitter_id].push_front(scent)
		else:
			active_scents_left[scent.smell_emitter_id] = [scent]
	elif origin.id == antenna_right.id:
		if active_scents_right.has(scent.smell_emitter_id):
			active_scents_right[scent.smell_emitter_id].push_front(scent)
		else:
			active_scents_right[scent.smell_emitter_id] = [scent]	
	else:
		print('origin unknown in add_scent!')

func remove_scent(origin, scent):
#	print('agent %s lost scent %s with signature %s' % [self, scent, scent.signature])
	
	if origin.id == antenna_left.id:
		if len(active_scents_left[scent.smell_emitter_id]) <= 1:
			active_scents_left.erase(scent.smell_emitter_id)
		else:
			var removed_scent = active_scents_left[scent.smell_emitter_id].pop_back()	
	elif origin.id == antenna_right.id:
		if len(active_scents_right[scent.smell_emitter_id]) <= 1:
			active_scents_right.erase(scent.smell_emitter_id)
		else:
			var removed_scent = active_scents_right[scent.smell_emitter_id].pop_back()		
	else:
		print('origin unknown in remove_scent!')		
	
func distance_from_scent(origin, scent):
	if scent == null:
		return null
		
	var distance = Globals.SMELL_DETECTABLE_RADIUS
	var detector_pos = origin.global_position
		
	if scent.is_inside_tree():
		var scent_pos = scent.global_position		
		distance = min(distance, detector_pos.distance_to(scent_pos))
	
	return distance
	
func get_combined_scent(origin, active_scents):
	var combined_scent_sig = Globals.NULL_SMELL

	for id in active_scents.keys():
		var scent = active_scents[id][-1]
		
		var distance = distance_from_scent(origin, scent)
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
func _on_smell_detected(origin, scent):
#	print('smell_detected from origin: ', origin)
	add_scent(origin, scent)

func _on_smell_lost(origin, scent):
#	print('smell_lost from origin: ', origin)
	remove_scent(origin, scent)

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
