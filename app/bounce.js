function setup(){
    createCanvas(640,360);
    pos = new mover(new pVector(100,100),new pVector(0,0),new pVector(0.2,0.2),0);
    //velocity = new mover(2,4);
}

function draw(){
    background(198);
    //pos.add(velocity);
    pos.update();
    //pos.position.log();
    //pos.accelleration.log();
    if((pos.position.x > width-8 && pos.accelleration.x > 0) || (pos.position.x < 8 && pos.accelleration.x < 0) ){
        pos.accelleration.x*=-1;
    }
    if((pos.position.y > height-8 && pos.accelleration.y > 0) || (pos.position.y < 8 && pos.accelleration.y < 0) ){
        pos.accelleration.y*=-1;
    }
    stroke(0);
    fill(275);
    ellipse(pos.position.x, pos.position.y, 16,16);
}