extends RigidBody2D

var signature = Globals.get_sensory_vector([0])

func _ready():
	# setting physics parameters
#	self.mass = 9.8
	self.linear_damp = 5.0
	self.gravity_scale = 0.0
	
	init_scent_areas([Globals.SMELL_DETECTABLE_RADIUS])

func init_scent_areas(radii):
	for r in radii:
		$SmellEmitter.add_scent_area(r, signature)	

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
