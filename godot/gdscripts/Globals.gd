# gdscript: globals.gd

extends Node

var RNG_SEED = 1 
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

const N_RANDOM_FOOD = 15
const N_RANDOM_OBS = 175
const N_AGENTS = 2

const WORLD_HORIZ_EXTENT = [0, 9000]
const WORLD_VERT_EXTENT = [-4100, 4500]

###################
# agent constants #
###################
const AGENT_INITIAL_HEALTH = 100
const AGENT_INITIAL_ENERGY = 50
const AGENT_INITIAL_SATIETY = 25

const AGENT_MAX_HEALTH = 100
const AGENT_MAX_ENERGY = 100
const AGENT_MAX_SATIETY = 100

# Time-based agent stats changes
const SATIETY_DECREASE_PER_FRAME = 0.1
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
const AGENT_MAX_SPEED_FORWARD = 300
const AGENT_MAX_SPEED_BACKWARD = 150

const AGENT_WALKING_ACCELERATION = 500
const AGENT_WALKING_FRICTION = 5000

const AGENT_MAX_ROTATION_RATE = 2.0

##########################
# agent action constants #
##########################
enum AGENT_ACTIONS {
	FORWARD,
	BACKWARD,
	TURN_LEFT,
	TURN_RIGHT
}

const MAX_PENDING_ACTIONS = 50

##############################################
# modifiable global state (USE WITH CAUTION) #
##############################################
var global_id = 0 # should never be acessed directly (use generate_unique_id)

# thread locks (mutexes)
var global_id_mutex = Mutex.new() # used in generate_unique_id function

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
