# GraphDitty

The purpose of this code is to compute clean weighted adjacency matrices for audio files using similarity network fusion, which can be used to do clean structure analysis.  This is based on ideas from the following papers:


<ol>
<li><a href = "http://www.ctralie.com">Christopher J Tralie</a> and <a href = "https://bmcfee.github.io/">Brian McFee</a>. "Enhanced Hierarchical Music Structure Annotations via Feature Level Similarity Fusion</a>.  International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2019.</li>

<li><a href = "http://www.ctralie.com">Christopher J Tralie</a>. <a href = "http://www.covers1000.net/ctralie2018_GraphDitty.pdf">GraphDitty: A Software Suite for Geometric Music Structure Visualization</a>.  In 19th International Society for Music Information Retrieval (ISMIR), 2018, Late Breaking Demo Session.</li>

<li><a href = "http://www.ctralie.com">Christopher J Tralie.</a> ``<a href = "http://www.covers1000.net/ctralie2017_EarlyMFCC_HPCPFusion.pdf">MFCC and HPCP fusion for robust cover song identification.''</a> In 18th International Society for Music Information Retrieval (ISMIR), 2017. </li>

</ol>

### Example Fused Adjacency Matrix on Michael Jacksons's Bad
<img src = "MJFusionExample.png" alt = "Example Adjacency Matrix on Michael Jacksons's Bad">


## Dependencies
* numpy/scipy/matplotlib
* [librosa]

## Running
To see all options, run the script as follows
~~~~~ bash
python SongStructure.py --help
~~~~~

There are options to view each iteration of the fusion.  By default, the results are saved to a .mat file called ``out.mat'' and to a file "out.json" (you can changes this via command line options). For example, the above figure was generated with the following call
~~~~~ bash
python SongStructure.py --filename MJ.mp3 --jsonfilename MJ.json
~~~~~

You can open the .json file in the web page "Viewer/index.html" to interactively view the similarity matrix and Laplacian eigenvectors, as well as as force graph and diffusion maps based on the similarity matrix.


[librosa]: <http://librosa.github.io/>
