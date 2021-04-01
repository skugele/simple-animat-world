# gdscript: smell-detector.gd

extends Node2D

signal smell_detected(origin, scent)
signal smell_lost(origin, scent)

onready var id = null

func _ready():
	id = Globals.generate_unique_id()
	
func enable():
	$Area2D/CollisionShape2D.disabled = false

func disable():
	$Area2D/CollisionShape2D.disabled = true

func _on_area_entered(scent):
#	print('entered: ', scent)
	emit_signal("smell_detected", self, scent)

func _on_area_exited(scent):
#	print('exited: ', scent)
	emit_signal("smell_lost", self, scent)
