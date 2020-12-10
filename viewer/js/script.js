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
        cleanButtons('buttonList')
        cleanButtons('gtButtonList')

        // generate jump buttons for items in the annotation
        genProgressButton(obj['annotation'], "buttonList")
        if (obj['gt_annotation'] != null) {
            genProgressButton(obj['gt_annotation'], "gtButtonList")
        }

        // load image
        var img = document.getElementById('ssmFigure');
        img.src = obj['figure']

        // image progress bar
        var player = document.getElementById('player');
        listener = function (evt) {
            var prg = document.getElementById("ssmProgress");
            v = player.currentTime / player.duration * 100;
            prg.style.backgroundColor = 'black'
            prg.style.height = '3px'
            prg.style.width = v + '%'
        }
        player.addEventListener('timeupdate', listener)
    }

    reader.readAsText(f);
}

function cleanButtons(id) {
    var ele = document.getElementById(id)
    while (ele.firstChild) {
        ele.removeChild(ele.firstChild)
    }
}

function buttonClz(label) {
    if (label.toLowerCase().startsWith("chorus")) {
        return 'button button2';
    } else {
        return 'button button1';
    }
}

var buttonListeners = {}
function genProgressButton(annotation, divID) {
    function getID(clz, index) {
        return divID + clz + index
    }

    var player = document.getElementById('player');
    annotation.forEach((item, index, array) => {
        var label = item['label'];
        var btn = document.createElement("button");
        btn.className = buttonClz(label)
        btn.id = getID('button', index);
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
        btnprogess.id = getID('progress', index);
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
        document.getElementById(divID).appendChild(btn);
    });

    // dynamically change button color to show position
    // capture annotation
    listener = ((annotation) => {
        return function (evt) {
            annotation.forEach((item, index, array) => {
                var begin = item['begin'];
                var end = item['end'];
                var btn = document.getElementById(getID('button', index));
                var prg = document.getElementById(getID('progress', index));
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
        }
    })(annotation);
    player.removeEventListener('timeupdate', buttonListeners[divID])
    buttonListeners[divID] = listener
    player.addEventListener('timeupdate', listener)
}

function changeAudioTime(t) {
    var player = document.getElementById('player');
    player.currentTime += t;
}