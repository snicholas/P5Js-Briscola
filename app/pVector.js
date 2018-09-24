class pVector{
    constructor(x_, y_) {
        this.x = x_;
        this.y = y_;
    }

    add(v){
        this.x += v.x;
        this.y += v.y;
    }

    substract(v){
        this.x -= v.x;
        this.y -= v.y;
    }

    multiply(s){
        this.x *= s;
        this.y *= s;
    }

    division(s){
        this.x /= s;
        this.y /= s;
    }
}