# gdscript: globals.gd

extends Node

var RNG_SEED = OS.get_ticks_msec()
var RNG = RandomNumberGenerator.new()

#############
# constants #
#############
const DEBUG = false

# UI constants
const TIME_FORMAT_STRING = '%02dD %02dH %02dM %02dS %03dms'

# layer bitmask values
const OBSTACLES_LAYER = 1
const AGENTS_LAYER = 2
const EDIBLES_LAYER = 4

# camera constants
const CAMERA_SMOOTHING_ENABLED = true
const CAMERA_SMOOTHING_SPEED = 2

###############################
# Environment Characteristics #
###############################
const RANDOMIZED = true

const N_RANDOM_FOOD = 50
const N_RANDOM_OBS = 175
const N_AGENTS = 0

const WORLD_HORIZ_EXTENT = [-15000, 15000]
const WORLD_VERT_EXTENT = [-12000, 12000]

###################
# agent constants #
###################
const AGENT_INITIAL_HEALTH = 100
const AGENT_INITIAL_ENERGY = 50
const AGENT_INITIAL_SATIETY = 250

const AGENT_MAX_HEALTH = 100
const AGENT_MAX_ENERGY = 100
const AGENT_MAX_SATIETY = 100000

# Time-based agent stats changes
const SATIETY_DECREASE_PER_FRAME = 1.0
const ENERGY_INCREASE_PER_FRAME = 0.75
const POISON_DECREASE_PER_FRAME = 0.01
const HEALTH_INCREASE_PER_FRAME = 0.1
const ENERGY_DECREASE_WHILE_HEALING_PER_FRAME = 0.25

const POISON_DAMAGE_PER_FRAME = 1.0
const STARVING_DAMAGE_PER_FRAME = 1.0

# Unit-based agent stats changes
const SATIETY_PER_UNIT_FOOD = 25.0
const ENERGY_PER_UNIT_FOOD = 2.0

const AGENT_ATTACK_DAMAGE = 20.0

############################
# agent movement constants #
############################
const AGENT_MAX_SPEED_FORWARD = 500
const AGENT_MAX_SPEED_BACKWARD = 500

const AGENT_WALKING_ACCELERATION = 20000
const AGENT_WALKING_FRICTION = 2000

const AGENT_MAX_ROTATION_RATE = 5.0

##########################
# agent action constants #
##########################
enum AGENT_ACTIONS {
	FORWARD=1,
	BACKWARD=2,
	TURN_LEFT=4,
	TURN_RIGHT=8
}

const NOOP_ACTION = 0
const MAX_PENDING_ACTIONS = 1

const SMELL_DIMENSIONS = 1
const SMELL_DETECTABLE_RADIUS = 5000.0
const SMELL_DISTANCE_MULTIPLIER = 5.0
const SMELL_DISTANCE_EXPONENT = 3.0

# base smells and tastes
var NULL_SMELL = get_sensory_vector([])

var SENSORS_STATE_TOPIC_FORMAT = '/agents/{agent_id}/sensors'

# identifiers for dictionary keys in outgoing observation messages
var OLFACTORY_SENSOR_ID = 'SMELL'
var TACTILE_SENSOR_ID = 'TOUCH'
var SOMATO_SENSOR_ID = 'SOMATOSENSORY'
var VELOCITY_ID = 'VELOCITY'
var LAST_ACTION_SEQNO_ID = 'LAST_ACTION'

##############################################
# modifiable global state (USE WITH CAUTION) #
##############################################
var global_id = -1 # should never be acessed directly (use generate_unique_id)
var agent_id = -1 # should never be acessed directly (use generate_unique_id)

# thread locks (mutexes)
var global_id_mutex = Mutex.new() # used in generate_unique_id function
var agent_id_mutex = Mutex.new() # used in generate_agent_id function

func _ready():
	randomize()
	seed(RNG_SEED)

##########################
# synchronized functions #
##########################
func generate_unique_id():
	var id = null
	
	# synchronized block
	global_id_mutex.lock()
	global_id += 1
	id = global_id
	global_id_mutex.unlock()

	return id

func generate_agent_id():
	var id = null
	
	# synchronized block
	agent_id_mutex.lock()
	agent_id += 1
	id = agent_id
	agent_id_mutex.unlock()

	return id

##################################################################
# shared functions (THESE FUNCTIONS SHOULD HAVE NO SIDE-EFFECTS) #
##################################################################
func get_elapsed_time():
	var milliseconds = (OS.get_ticks_msec())
	
	var remainder = 0
	
	var days = milliseconds / 86400000
	remainder = milliseconds % 86400000
	
	var hours = remainder / 3600000
	remainder = remainder % 3600000
	
	var minutes = remainder / 60000
	remainder = remainder % 60000
	
	var seconds = remainder / 1000
	milliseconds = remainder % 1000	
	
	return [days, hours, minutes, seconds, milliseconds]

# assume vectors have same length
func add_vectors(v1, v2):
	var result = v1.duplicate()
	
	var i = 0
	while i < len(v1):
		result[i] += v2[i]
		i += 1
		
	return result		

func add_vector_array(array):
	if array == null or len(array) == 0:
		return null
		
	var result = array.pop_front()
	for v in array:
		result = add_vectors(result, v)
		
	return result		
	
func zero_vector(dims):
	var new_vector = []
	for d in dims:
		new_vector.append(0)
		
	return new_vector

# "scales" an array by some scalar value
func scale(array, scalar):
	if array == null or len(array) == 0:
		return array
		
	var new_array = array.duplicate()
	
	var pos = 0
	while pos < len(new_array):
		new_array[pos] *= scalar
		pos += 1
		
	return new_array	
	
func get_magnitude(vector):
	var value = 0.0
	
	for element in vector:
		value += pow(element, 2)
		
	return sqrt(value)
	
func normalize(vector):
	var magnitude = get_magnitude(vector)
	if magnitude == 0:
		return vector		
		
	return	scale(vector, 1.0 / magnitude)
	
func get_sensory_vector(active_elements):
	var vector = []
	
	# check for active and set to 1 else 0
	var i = 0
	while i < SMELL_DIMENSIONS:
		
		# not active
		if active_elements.find(i) == -1:
			vector.append(0)
			
		# active
		else:
			vector.append(1)
			
		i += 1
				
	# normalize vector
	return normalize(vector)
