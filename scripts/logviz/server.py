#!/usr/bin/python

from bottle import route, run
# to install bottle package locally run:
#   pip install --user bottle
import subprocess

@route('/logviz/<file_path:path>')
def log(file_path):
    # root for log files
    root = '/home/USER/models-log/'
    # script path prefix
    pref = '/home/USER/oxnn/scripts/logviz/'

    script = pref + 'log_dygraphs_csv.lua'
    script_annot = pref + 'log_dygraphs_annotations.lua'
    script_time = pref + 'log_dygraphs_time.lua'
    output = '''
    <html>
        <head>
            <title>oxnn - logviz</title>
            <script src="//cdnjs.cloudflare.com/ajax/libs/dygraph/1.1.0/dygraph-combined.js"></script>
        </head>
        <body>
        <div id="graphdiv" style="width: 100%; height: 100%;"></div>
        <div id="graphdivtime" style="width: 100%; height: 30%;"></div>
        <script type="text/javascript">
          g = new Dygraph(

            // containing div
            document.getElementById("graphdiv"),

            // CSV or path to a CSV file.
    '''
    #output = output +
    out = subprocess.check_output(['luajit', script, root+file_path])
    output = output + '"'
    for line in out.splitlines():
        output = output  + line + '\\n'
    output = output + '"\n'
    output = output + '''
              ,
            {
                title: 'Perplexity at Time',
                connectSeparatedPoints: true,
                drawPoints: true,
                legend: 'always',
            }
          );
          g.ready(function() {
            g.setAnnotations(['''
    out = subprocess.check_output(['luajit', script_annot, root+file_path])
    for line in out.splitlines():
        output = output  + line + '\n'
    output = output + '''
            ]);
          });
          g = new Dygraph(

            // containing div
            document.getElementById("graphdivtime"),

            // CSV or path to a CSV file.
    '''
    out = subprocess.check_output(['luajit', script_time, root+file_path])
    output = output + '"'
    for line in out.splitlines():
        output = output  + line + '\\n'
    output = output + '"\n'
    output = output + '''
              ,
            {
                title: 'Seconds per Minibatch',
                connectSeparatedPoints: true,
                drawPoints: true,
                legend: 'always',
            }
          );
    </script>
    <div>'''
    with open(root+file_path) as f:
        output = output + f.read().replace('\n', '<br/>\n')
    output = output + '''</div>
    </body>
    </html>'''

    return output

run(host='0.0.0.0', port=8080)
