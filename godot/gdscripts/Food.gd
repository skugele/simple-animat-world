extends RigidBody2D


func _ready():
	# setting physics parameters
#	self.mass = 9.8
	self.linear_damp = 5.0
	self.gravity_scale = 0.0


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
