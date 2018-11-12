class Mover{
	constructor(position_, velocity_, acceleration_, type_, mass_, force_){
		this.position = position_;
		this.velocity = velocity_;
		this.acceleration = acceleration_;
		this.type = type_;
		this.mass = mass_;
		this.force = force_;
	}

	update(){
		var force = new pVector(this.force.x, this.force.y);
		this.applyForce(force);
		this.velocity.add(this.acceleration);
		this.velocity.limit(5);
		this.position.add(this.velocity);

	}
	applyForce(force) {
		var f = new pVector(force.x, force.y);
		f.division(this.mass);
		this.acceleration.add(force);
	}
	
	display() {
		stroke(0);
		fill(175);
		ellipse(position.x, position.y, 16,16);
	}


}