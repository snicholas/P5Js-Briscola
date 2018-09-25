class mover{
	constructor(position_, velocity_, accelleration_, type_){
		this.position = position_;
		this.velocity = velocity_;
		this.accelleration = accelleration_;
		this.type = type_;
	}

	update(){
		var dir = new pVector(random(width),random(height));
		dir.normalize();
		dir.multiply(0.5);
		this.velocity.add(this.accelleration);
		this.velocity.limit(2);
		this.position.add(this.velocity);

	}

	display() {
		stroke(0);
		fill(175);
		ellipse(position.x, position.y, 16,16);
	}


}