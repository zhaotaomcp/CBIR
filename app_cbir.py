#! /usr/bin/env python
#coding=utf-8

import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import sys
sys.path.append("/opt/down/caffe-master/python")
import caffe

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/examples/cbir/model/cbir.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/examples/cbir/model/cbir_iter_100000.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/examples/cbir/data/imagenet_mean.npy'.format(REPO_DIRNAME)),
        #'class_labels_file': (
        #    '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        #'lsh_file': (
        #    '{}lsh10.pkl'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )
        """
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values
        
        self.bet = cPickle.load(open(lsh_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1
        """

        #load nearly index
        lsh_file = open('lsh10.pkl', 'rb')
        self.lsh = cPickle.load(lsh_file)
        lsh_file.close()

        #get candidate images by query result
        index_file = open('index10_fea.pkl', 'rb')
        self.index_data = cPickle.load(index_file)
        index_file.close()

        fea_file = open('img10_fea.pkl', 'rb')
        self.fea_data = cPickle.load(fea_file)
        fea_file.close()

	train_file = open('static/train.txt', 'rb')
	self.train_txt = train_file.readlines()
	train_file.close()


    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            # get fc7 layer features
            fc7_fea = self.net.blobs["fc7"].data[:]
            logging.info('fc7_fea:%d,%d',fc7_fea.shape[0], fc7_fea.shape[1])

            # get fc8 layer features
            fc8_fea = self.net.blobs["fc8_encode"].data[:]
            fc8_fea = (fc8_fea>=0.5)*1
            logging.info('fc8_fea:%d,%d',fc8_fea.shape[0], fc8_fea.shape[1])

            #search nearly index
            raw_result = self.lsh.query(np.array([int(tmp) for tmp in fc8_fea[0]]))
            print 'lsh.query result:'
            print raw_result

            candidate_dic = {}

            vec1 = np.array([int(tmp*100) for tmp in fc7_fea[0]])
            print 'vec1:'
            print vec1

            for x in raw_result:
                    print 'x:'
                    print x
                    index_lsh = ','.join(str(i) for i in x[0]).replace(',', '')
                    print 'the candidate image id is:'
                    print index_lsh
                    print 'the candidate image name is:'
                    print self.index_data[index_lsh]

                    for index in self.index_data[index_lsh]:
                            print 'the candidate image feature is:'
                            condidate_fea = self.fea_data[''.join(index)]
                            print condidate_fea.toarray()

                            vec2 = np.array([int(tmp*100) for tmp in condidate_fea.toarray()[0]])
                            print 'vec2'
                            print vec2
                            dist = np.linalg.norm(vec1 - vec2)
                            print 'distance:'
                            print int(dist)

                            candidate_dic[''.join(index)] = int(dist)

            candidate_list = sorted(candidate_dic.items(), key=lambda item:item[1])
            print 'candidate_list:'
            print candidate_list

	    # Get file path from candidate list
	    i = 0
	    for candidate_item in candidate_list:
	        candidate_str = candidate_item[0][0:-4] + '_' + candidate_item[0][-4:]
	        print 'candidate_str:'+candidate_str
	        for line in self.train_txt:
	            filename, tag = line.strip().split(" ")
		    #print 'filename:'+filename
	            if candidate_str in filename:
	                candidate_list[i] = ("/static/"+filename, candidate_list[i][1])
	                i += 1
	                break

	    print candidate_list
	                


            return (True, candidate_list, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
