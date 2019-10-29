var deck = [];
var statodeck = [];
var statuses = ['indeck', 'p1hand', 'p1played', 'p2hand', 'p2played',
    'p1taken', 'p2taken'
];
var briscola = null;
var p1, p2;
var curPlayer = 1;
var played = 0;
var p1Played, p2Played;
var finished = false;
var wait = false;



function setup() {
    // console.log(statodeck)
    createCanvas(400, 400);
    background(220);
    for (j = 0; j < 4; j++) {
        for (i = 1; i <= 10; i++) {
            var c = new Card(i + (j * 10), i, j + 1);
            deck.push(c);
            statodeck.push(statuses[0]);
        }
    }
    shuffle(deck, true);
    briscola = deck[38];
    p1 = new Player(false, 1);
    p2 = new Player(false, 2);
    p1.cards = deck.slice(0, 3);
    deck = deck.slice(3);
    p2.cards = deck.slice(0, 3);
    deck = deck.slice(3);
    curPlayer = p1.id;
    for (var i = 0; i < p1.cards.length; i++) {
        statodeck[p1.cards[i].id - 1] = statuses[1];
    }
    for (var i = 0; i < p2.cards.length; i++) {
        statodeck[p2.cards[i].id - 1] = statuses[3];
    }
}

function mouseClicked() {
    if (p1.ishuman && curPlayer === p1.id) {
        for (var i = 0; i < p1.cards.length; i++) {
            if (p1.cards[i].x < mouseX && mouseX < p1.cards[i].x + 50) {
                if (p1.cards[i].y < mouseY && mouseY < p1.cards[i].y + 90) {
                    p1.selectedCard = p1.cards[i];
                    p1Played = p1.selectedCard;
                    p1.cards[i].chosen = true;
                    played++;
                    curPlayer = p2.id;
                    statodeck[p1.cards[i].id - 1] = statuses[2];
                }
            }
        }
    }
    if (p2.ishuman && curPlayer === p2.id) {
        for (var i = 0; i < p2.cards.length; i++) {
            if (p2.cards[i].x < mouseX && mouseX < p2.cards[i].x + 50) {
                if (p2.cards[i].y < mouseY && mouseY < p2.cards[i].y + 90) {
                    //p2Played = p2.playCard(p2.cards[i],briscola.seme,0);
                    p2.selectedCard = p1.cards[i];
                    p2Played = p1.selectedCard;
                    p2.cards[i].chosen = true;
                    played++;
                    curPlayer = p1.id;
                    statodeck[p1.cards[i].id - 1] = statuses[4];
                }
            }
        }
    }
}

function draw() {
    background(220);
    if (!finished) {
        // console.log(statodeck);
        if (!p2Played && p2.cards.length > 0) {
            p2.cards.forEach(element => {
                element.chosen = false;
            });
        } else if (p2Played && p2.cards.length > 0) {
            p2.cards.forEach(element => {
                element.chosen = element.id == p2Played.id;
            });
        }
        if (!p1Played && p1.cards.length > 0) {
            p1.cards.forEach(element => {
                element.chosen = false;
            });
        } else if (p1Played && p1.cards.length > 0) {
            p1.cards.forEach(element => {
                element.chosen = element.id == p1Played.id;
            });
        }
        y = 20;
        x = 20;
        for (var i = 0; i < p2.cards.length; i++) {
            p2.cards[i].display(true, x, y);
            x += 60;
        }
        y = 200;
        x = 20;
        for (var i = 0; i < p1.cards.length; i++) {
            p1.cards[i].display(true, x, y);
            x += 60;
        }
        if (deck.length > 0) {
            briscola.display(true, 300, 100);
        }
        textSize(14);
        text("P1 score: " + p1.score, 300, 20);
        text("P2 score: " + p2.score, 300, 50);
        if (wait) {
            noLoop();
            setTimeout(function() {
                loop();
                wait = false;
            }, 1000);
        } else {
            if (played >= 2) {
                played = 0;
                winner(p1Played, p2Played);
                wait = true;
                p2Played = null;
                p1Played = null;
                if (deck.length == 0 && p1.cards.length == 0 && p2.cards.length == 0) {
                    // console.log('End!');
                    finished = true;
                } else if (deck.length > 0) {
                    var c = deck.slice(0, 2);
                    if (curPlayer === p1.id) {
                        p1.cards.push(c[0]);
                        p2.cards.push(c[1]);
                        statodeck[c[0].id - 1] = statuses[1];
                        statodeck[c[1].id - 1] = statuses[3];
                    }
                    if (curPlayer === p2.id) {
                        p2.cards.push(c[0]);
                        p1.cards.push(c[1]);
                        statodeck[c[0].id - 1] = statuses[3];
                        statodeck[c[1].id - 1] = statuses[1];
                    }
                    deck = deck.slice(2);
                }
            } else {
                if (curPlayer === p1.id && !p1.ishuman && !p1.isthinking) {
                    var cavv = 0;
                    if (p2Played) {
                        cavv = p2Played.id;
                    }
                    p1.isthinking = true;
                    //p1.playCard(null, briscola.seme, cavv);
                    p1.playCard(statodeck,briscola.seme)
                } else if (curPlayer === p1.id && !p1.ishuman && p1.selectedCard) {
                    p1Played = p1.selectedCard;
                    statodeck[p1.selectedCard.id - 1] = statuses[2];
                    p1.selectedCard = null;
                    p1.isthinking = false
                    played++;
                    curPlayer = p2.id;
                    wait = true;
                } else if (curPlayer === p2.id && !p2.ishuman && !p2.isthinking && !p2.selectedCard) {
                    p2.isthinking = true;
                    p2.selectedCard = null;
                    var cavv = 0;
                    if (p1Played) {
                        cavv = p1Played.id;
                    }
                    //p2.playCard(null, briscola.seme, cavv);
                    p2.playCard(statodeck,briscola.seme)
                } else if (curPlayer === p2.id && !p2.ishuman && p2.selectedCard) {
                    p2Played = p2.selectedCard;
                    statodeck[p2.selectedCard.id - 1] = statuses[4];
                    p2.selectedCard = null;
                    p2.isthinking = false;
                    played++;
                    curPlayer = p1.id;
                    wait = true;
                }
            }
        }


    } else {
        textSize(14);
        text("P1 score: " + p1.score, 300, 20);
        text("P2 score: " + p2.score, 300, 50);
        textSize(24);
        if (p1.score > p2.score) {
            text("P1 Wins!!!", 100, 100);
        } else if (p1.score < p2.score) {
            text("P2 Wins!!!", 100, 100)
        } else {
            text("Draw!!!", 100, 100)
        }
        noLoop();
    }
}

function winner(c1, c2) {
    console.log('-----------------------');
    c1.log();
    c2.log();
    console.log('curPlayer', curPlayer);
    console.log('_______________________');
    played = 0;
    if (c1.seme == briscola.seme && c2.seme != briscola.seme) {
        p1.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[5];
        statodeck[c2.id - 1] = statuses[5];
        curPlayer = p1.id;
    } else if (c2.seme == briscola.seme && c1.seme != briscola.seme) {
        p2.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[6];
        statodeck[c2.id - 1] = statuses[6];
        curPlayer = p2.id;
    } else if (c1.value > c2.value && c1.seme === c2.seme) {
        p1.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[5];
        statodeck[c2.id - 1] = statuses[5];
        curPlayer = p1.id;
    } else if (c2.value == c1.value && c1.seme === c2.seme && c2.numuero > c1.numero) {
        p2.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[6];
        statodeck[c2.id - 1] = statuses[6];
        curPlayer = p2.id;
    } else if (c1.value > c2.value && c1.seme === c2.seme && c1.numuero > c2.numero) {
        p1.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[5];
        statodeck[c2.id - 1] = statuses[5];
        curPlayer = p1.id;
    } else if (c2.value > c1.value && c1.seme === c2.seme) {
        p2.score += (c1.value + c2.value);
        statodeck[c1.id - 1] = statuses[6];
        statodeck[c2.id - 1] = statuses[6];
        curPlayer = p2.id;
    } else {
        if (curPlayer === p1.id) {
            p1.score += (c1.value + c2.value);
            statodeck[c1.id - 1] = statuses[5];
            statodeck[c2.id - 1] = statuses[5];
            curPlayer = p1.id;
        }
        if (curPlayer === p2.id) {
            p2.score += (c1.value + c2.value);
            statodeck[c1.id - 1] = statuses[6];
            statodeck[c2.id - 1] = statuses[6];
            curPlayer = p2.id;
        }
    }
    p1.cards.splice(p1.cards.indexOf(c1), 1);
    p2.cards.splice(p2.cards.indexOf(c2), 1);
    console.log('turn: P' + curPlayer);
    console.log('-----------------------');
}