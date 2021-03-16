# gdscript: smell-emitter.gd

extends Node2D

onready var id = null

func _ready():
	id = Globals.generate_unique_id()
		
func add_scent_area(radius):
	var scent_area = load("res://scenes/ScentArea.tscn").instance()
	
	scent_area.init_shape(radius)
	scent_area.smell_emitter_id = id
#
	$ScentAreas.add_child(scent_area)
	
	return scent_area
