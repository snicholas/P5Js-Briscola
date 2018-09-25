class pVector {
    constructor(x_, y_) {
        this.x = x_;
        this.y = y_;
    }
    log() {
        console.log('x: ' + this.x, 'x: ' + this.y, 'mag: ' + this.mag());
    }
    add(v) {
        this.x += v.x;
        this.y += v.y;
    }

    substract(v) {
        this.x -= v.x;
        this.y -= v.y;
    }

    multiply(s) {
        this.x *= s;
        this.y *= s;
    }

    division(s) {
        if (s != 0) {
            this.x /= s;
            this.y /= s;
        }
    }

    mag() {
        return sqrt(this.x * this.x + this.y * this.y);
    }

    normalize() {
        this.division(this.mag())
    }

    limit(v){
        var m=this.mag();
        if(m>v){
            this.division(m/v);
        }
    }
}