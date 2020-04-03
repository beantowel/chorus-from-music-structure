function handleFileSelect(evt) {
    var files = evt.target.files; // FileList object
    var f = files[0];
    var reader = new FileReader();

    reader.onload = (function (theFile) {
        return function (evt) {
            // list some file properties.
            var output = ['<li><strong>', f.name, '</strong> (', f.type || 'n/a', ') - ',
                f.size, ' bytes, last modified: ',
                f.lastModifiedDate ? f.lastModifiedDate.toLocaleDateString() : 'n/a',
                '</li>'].join('');
            document.getElementById('echo').innerHTML = '<ul>' + output + '</ul>';
        }
    })(f);

    reader.onloadend = function (evt) {
        // display audio title
        var txt = evt.target.result;
        var obj = JSON.parse(txt);
        var title = document.getElementById('audioTitle');
        title.innerHTML = '<ul>' + obj['audio'] + '</ul>';
        title.href = obj['audio'];

        // load audio to player
        var player = document.getElementById('player');
        document.getElementById('playerSource').src = obj['audio'];
        player.load();

        // clear previous buttons
        var ele = document.getElementById('buttonList');
        while (ele.firstChild) {
            ele.removeChild(ele.firstChild);
        }

        // generate jump buttons for items in the annotation
        obj['annotation'].forEach((item, index, array) => {
            var label = item['label'];
            var btn = document.createElement("button");
            btn.className = buttonClz(label)
            btn.id = 'button' + index;
            btn.innerHTML = label;
            btn.style.width = Math.max(105, 5 * (item['end'] - item['begin'])) + 'px'

            // progress bar inside button
            var prg = document.createElement('div');
            prg.style.width = '100%';
            prg.style.backgroundColor = '#EDEDED';
            var btnprogess = document.createElement('div');
            btnprogess.style.width = '0%';
            btnprogess.style.height = '2px'
            btnprogess.style.backgroundColor = 'black';
            btnprogess.id = 'button progress' + index;
            btn.appendChild(prg);
            prg.appendChild(btnprogess);

            // capture current item
            btn.onclick = ((item) => {
                return function (evt) {
                    // jump to annotation beginning
                    var begin = item['begin'];
                    var end = item['end'];
                    player.currentTime = begin;
                    player.play();
                    // stopAt(player, end);
                }
            })(item);
            document.getElementById('buttonList').appendChild(btn);
        });

        // dynamically change button color to show position
        player.addEventListener('timeupdate', (evt) => {
            obj['annotation'].forEach((item, index, array) => {
                var begin = item['begin'];
                var end = item['end'];
                var btn = document.getElementById('button' + index);
                var prg = document.getElementById('button progress' + index);
                if (player.currentTime > begin && player.currentTime < end - 1e-4) {
                    btn.className = 'button button3';
                    // show segment progress
                    v = (player.currentTime - begin) / (end - begin) * 100;
                    prg.style.height = '3px'
                    prg.style.width = v + '%'
                } else {
                    btn.className = buttonClz(item['label'])
                    // clear segment progress
                    prg.style.height = '0px'
                }
            })
        });
    }

    reader.readAsText(f);
}

function buttonClz(label) {
    if (label.toLowerCase().startsWith('chorus')) {
        return 'button button2';
    } else {
        return 'button button1';
    }
}

var listeners = {}
function stopAt(player, end) {
    player.removeEventListener('timeupdate', listeners[player.id])
    function stopOnce(evt) {
        if (player.currentTime > end) {
            player.pause();
            player.removeEventListener('timeupdate', listeners[player.id]);
        }
    }
    player.addEventListener('timeupdate', stopOnce);
    listeners[player.id] = stopOnce;
}