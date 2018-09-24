function setup(){
    createCanvas(640,360);
    pos = new pVector(100,100);
    velocity = new pVector(2.5,5);
}

function draw(){
    background(255);
    pos.add(velocity);
    if((pos.x > width) || (pos.x < 0) ){
        velocity.x*=-1;
    }
    if((pos.y > height) || (pos.y < 0) ){
        velocity.y*=-1;
    }
    stroke(0);
    fill(275);
    ellipse(pos.x, pos.y, 16,16);
}