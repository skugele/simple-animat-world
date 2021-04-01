extends KinematicBody2D

onready var has_camera = false

# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass

func set_camera(camera):
	if not has_camera:
		add_child(camera)
		has_camera = true
		
func unset_camera(camera):
	if has_camera:
		remove_child(camera)
		has_camera = false
	
func move_up():
	global_position += Vector2(0, -100)
	
func move_down():
	global_position += Vector2(0, 100)
	
func move_left():
	global_position += Vector2(-100, 0)
	
func move_right():
	global_position += Vector2(100, 0)
