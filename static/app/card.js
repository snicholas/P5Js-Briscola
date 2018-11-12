var semi = {
    1: 'cuori',
    2: 'quadri',
    3: 'fiori',
    4: 'picche'
}
class Card {
    constructor(id, numero, seme) {
        this.numero = numero;
        this.seme = seme;
        this.id = id;
        this.value = 0;
        this.x = 0;
        this.y = 0;
        this.chosen=false;
        this.name=this.numero + ' '+ semi[this.seme];
        if (this.numero == 1) {
            this.value = 11;
        }
        else if (this.numero == 3) {
            this.value = 10;
        }
        else if (this.numero == 8) {
            this.value = 2;
        }
        else if (this.numero == 9) {
            this.value = 3;
        }
        else if (this.numero == 10) {
            this.value = 4;
        }
    }
    log(){
        console.log(this.name + ' '+ this.value);
    }
    display(visible, x, y) {
        this.x = x;
        this.y = y;
        var w = 50;
        var h = 90;
        if(this.chosen){
            w+=20;
            h+=20;
        }
        fill(255);
        rect(x, y, w, h, 20);
        fill(0);
        textSize(10);
        text(this.numero, x + 10, y + 40);
        text(semi[this.seme], x + 10, y + 60);
    }


}