class Walker {
    constructor(width, height) {
        this.x = parseInt(width / 2);
        this.y = parseInt(height / 2);
        this.tx = 0;
        this.ty = 10000;
    };
    display() {
        noStroke();
        fill(0, 20);
        ellipse(this.x, this.y, 10, 10);
    }
    step() {
        this.x = map(noise(this.tx),0,1,0,400);//randomGaussian(0.05, .2); //random(-1, 1);
        this.y = map(noise(this.ty),0,1,0,400);
        //this.y -= randomGaussian(0.1,.3);//random(-1, 1);
        this.tx+=0.01;
        this.ty+=0.01;
        //console.log(this.x, this.y);
    };
}