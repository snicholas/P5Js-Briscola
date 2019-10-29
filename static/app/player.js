class Player {
    constructor(human, id) {
        this.ishuman = human;
        this.id = id;
        this.cards = [];
        this.score = 0;
        this.isthinking = false;
        this.selectedCard = null;
        this.playCard = function (deck, seed) {
            let self = this;
            self.isthinking = true;
            // if (cavv == null) {
            //     cavv = 0;
            // }
            if (!self.ishuman) {
                // let c1 = self.cards[0].id;
                // let c2 = 0;
                // let c3 = 0;
                // if (self.cards.length > 1) {
                //     c2 = self.cards[1].id;
                //     if (self.cards.length == 3) {
                //         c3 = self.cards[2].id;
                //     }
                // }
                // let dt = {
                //     "c1": c1,
                //     "c2": c2,
                //     "c3": c3,
                //     "ca": cavv,
                //     "briscola": briscolaSeme,

                // }
                let c = -1;
                postData('http://127.0.0.1:5000/briscola/playcardV2', {deck:deck, player:this.id, seed})
                .then(function (v) {
                    c = v;
                    let tc = self.cards.find(function (cc) { return cc.id == c; });
                    var idx = -1;
                    if (tc) {
                        self.cards.forEach((el,ix) => {
                            if(el.id==tc.id){
                                idx=ix;
                            }
                        });
                        self.selectedCard = tc;
                    } else {
                        idx=floor(random(self.cards.length));
                        self.selectedCard = self.cards[idx];
                    }
                    self.isthinking = false;
                    if(idx!=-1 && self.cards.length>0){
                        self.cards[idx].chosen = true;
                    }
                })
            }

        }
    }
}
async function postData(url = ``, data = {}) {
    // Default options are marked with *
    return await fetch(url, {
        method: "POST", // *GET, POST, PUT, DELETE, etc.
        mode: "cors", // no-cors, cors, *same-origin
        cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
        credentials: "same-origin", // include, *same-origin, omit
        headers: {
            "Content-Type": "application/json; charset=utf-8",
            // "Content-Type": "application/x-www-form-urlencoded",
        },
        redirect: "follow", // manual, *follow, error
        referrer: "no-referrer", // no-referrer, *client
        body: JSON.stringify(data), // body data type must match "Content-Type" header
    })
        .then(response => response.json()); // parses response to JSON
}