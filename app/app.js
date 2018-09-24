var w;
function setup() {
  createCanvas(400, 400);
  background(220);
  w = new Walker(400, 400);
  w.display();
}

function draw() {
  
  w.step();
  w.display();

}