function setup(){
    createCanvas(640,360);
    pos = new Mover(new pVector(100,100),new pVector(0,0),new pVector(0.01,0.02),0, 10, new pVector(0.01,0.01));
    //velocity = new mover(2,4);
}

function draw(){
    background(198);
    //pos.add(velocity);
    pos.update();
    //pos.position.log();
    //pos.accelleration.log();
    if((pos.position.x > width-8 && pos.acceleration.x > 0) || (pos.position.x < 8 && pos.acceleration.x < 0) ){
        pos.acceleration.x=0;
        pos.force.x*=-1;
        pos.velocity.x = 0.01;
    }
    if((pos.position.y > height-8 && pos.acceleration.y > 0) || (pos.position.y < 8 && pos.acceleration.y < 0) ){
        pos.acceleration.y = 0;
        pos.force.y*=-1;
        pos.velocity.y = 0.01;
    }
    pos.velocity.multiply(0.99);
    stroke(0);
    fill(275);
    ellipse(pos.position.x, pos.position.y, 16,16);
}