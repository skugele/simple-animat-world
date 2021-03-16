extends Node2D

signal edible_consumed(edible)

func _ready():
	pass

func _on_edible_encountered(edible):
	emit_signal("edible_consumed", edible)
