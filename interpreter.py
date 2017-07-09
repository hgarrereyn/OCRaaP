import numpy as np
import cv2
import image_parser
import model
import os
import sys
import getopt

# Utility methods

def dist2(x,y,px,py):
    return ((px - x) * (px - x)) + ((py - y) * (py - y))

def ang(x,y,px,py):
    return bound_ang(np.angle(np.complex(x-px, y-py)))

def bound_ang(theta):
    theta %= (np.pi * 2)

    if theta > np.pi:
        theta -= (np.pi * 2)

    return theta

def title(p):
    return ['sad', 'dead', 'at', 'hash', 'conf', 'empty', 'dot', 'dollar', 'plus', 'dash'][p]

def s_title(p):
    return ['S', 'D', '@', '#', '?', 'O', '.', '$', '+', '-'][p]

# program class

class Program:

    def __init__(self, source, debug=False, quiet=False):
        self.debug = debug
        self.quiet = quiet

        # make tensorflow quiet
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # welcome message
        self.say('OCRaaP Interpreter v1.0\n')

        # load the trained classifier
        self.say('* Loading classifier')
        self.model = model.Model(fromCheckpoint='milestones/classifier_v1.0.ckpt')

        # -- setup display images
        self.err_image = cv2.imread(source)
        self.err_image = cv2.resize(self.err_image, (1000,1000), None)

        self.disp_image = cv2.imread(source)
        self.disp_image = cv2.resize(self.disp_image, (1000,1000), None)

        if self.debug:
            self.show('Source', self.disp_image)

        # -- parse time variables

        # id: (x, y, p, s)
        self.sym_list = []

        # -- end

        # -- Runtime variables

        # id of start symbol
        self.r_start = None

        # id of current symbol
        self.r_pointer = None

        # id of previous symbol
        self.r_prev_pointer = None

        # angle of pointer direction
        self.r_pointer_dir = None

        # the value stack
        self.r_stack = []

        # 0 - normal
        # 1 - value load
        self.r_mode = 0

        # for constant loading
        self.r_const = 0
        self.r_const_i = 0

        # conf sets to true
        self.r_force_launch = False

        # -- end

        # parse the source image
        self.parseSource(source)

    def say(self, message):
        if not self.quiet:
            print(message)

    def dbg(self, message):
        if self.debug:
            print(message)

    def show(self, name, im):
        small = cv2.resize(im, (600,600), None)
        cv2.imshow(name, small)
        return cv2.waitKey(0)

    def err(self, code, reason, loc, theta=None, disp=True):
        print('\nERROR [' + str(code) + ']:')
        print('- ' + reason)

        if disp:
            for l in loc:
                x,y,p,s = self.sym_list[l]
                cv2.circle(self.err_image, (x, y) , s, (0,0,255), 2)
                print('@ (' + str(x) + ', ' + str(y) + ')')

            if theta:
                px = int(x + np.cos(theta) * 70)
                py = int(y + np.sin(theta) * 70)
                cv2.line(self.err_image, (x, y), (px, py), (0,0,255), 2)

            self.show('Error [' + str(code) + ']', self.err_image)

        sys.exit(1)

    def parseSource(self, source):
        self.dbg('* Loading source: [' + source + ']')

        # read the image
        prog = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

        # fetch & predict symbols
        symbols, keypoints = image_parser.get_symbols(prog)
        raw = self.model.run_predict(symbols)
        predictions = np.argmax(raw[0], 1)

        # loop through the symbols and store symbols
        for k in range(len(keypoints)):
            x = int(keypoints[k].pt[0])
            y = int(keypoints[k].pt[1])
            p = predictions[k]
            s = int(keypoints[k].size * 1.2)

            # storage
            self.sym_list.append((x,y,p,s))

            i = len(self.sym_list) - 1

            # look for the start point
            if p == 0:
                # Error: duplicate start point
                if self.r_start != None:
                    self.err(10, 'Too sad', [self.r_start, i])
                else:
                    self.r_start = i

            # debug info
            self.dbg('\t[' + str(int(x)) + ', ' + str(int(y)) + ', ' + title(p) + ']')

        # Error: no start point
        if self.r_start == None:
            self.err(11, 'Too happy', [])


    def run(self, maxIterations=10000):
        self.dbg('* Running')

        x,y,_,_ = self.sym_list[self.r_start]

        self.r_pointer = self.r_start
        self.r_pointer_dir = 0

        for i in range(maxIterations):
            o = self.step(i)

            if o == -1:
                return

    def step(self,i):
        # fetch the next symbol
        sym = self.nextSymbol()
        x,y,p,_ = self.sym_list[sym]

        # process symbol
        o = self.processSymbol(p, sym)

        # debugging info
        self.dbg('[' + str(i) + '](' + str(x).zfill(3) + ', ' + str(y).zfill(3) + '){' + s_title(p) +'} :: ' + str(self.r_stack))

        if self.debug:
            # debug view
            di = self.disp_image.copy()

            x,y,p,s = self.sym_list[self.r_pointer]
            cv2.circle(di, (x, y) , s, (255,0,0), 2)

            px = int(x + np.cos(self.r_pointer_dir) * 70)
            py = int(y + np.sin(self.r_pointer_dir) * 70)

            cv2.line(di, (x, y), (px, py), (255,0,0), 2)

            self.show('Debug', di)

        return o

    # Do the thing
    def processSymbol(self, p, sym):
        self.r_force_launch = False

        x,y,p,_ = self.sym_list[sym]
        xp,yp,_,_ = self.sym_list[self.r_pointer]
        theta = ang(x,y,xp,yp)

        self.r_prev_pointer = self.r_pointer
        self.r_pointer = sym
        self.r_pointer_dir = theta

        # sad (start)
        if p == 0:
            self.err(21, 'already sad', [sym])

        # dead (end)
        elif p == 1:
            self.say('* Done')
            return -1

        # at (read)
        elif p == 2:
            v = ord(sys.stdin.read(1))
            if v == ord('\n'):
                v = 0
            self.push(v)

        # hash (write)
        elif p == 3:
            v = self.pop()
            sys.stdout.write(chr(v))

        # conf
        elif p == 4:
            v = self.pop()
            self.r_force_launch = True

            if v == 0:
                self.r_pointer_dir -= (np.pi / 3)
            else:
                self.r_pointer_dir += (np.pi / 3)

            self.r_pointer_dir = bound_ang(self.r_pointer_dir)

        # empty
        elif p == 5:
            self.const_load(0)

        # dot
        elif p == 6:
            self.const_load(1)

        # dollar
        elif p == 7:
            # duplicate
            if self.r_mode == 0:
                v = self.pop()
                self.push(v)
                self.push(v)
            # load const
            else:
                self.r_mode = 0
                self.push(self.r_const)

        # plus
        elif p == 8:
            b = self.pop()
            a = self.pop()
            self.push(a + b)

        # dash
        elif p == 9:
            b = self.pop()
            a = self.pop()
            self.push(a - b)


    def pop(self):
        if len(self.r_stack) > 0:
            v = self.r_stack.pop()
            return v
        else:
            self.err(20, 'Stack underflow', [self.r_pointer])

    def push(self, v):
        self.r_stack.append(v)

    def const_load(self,val):
        if self.r_mode == 0:
            self.r_mode = 1
            self.r_const = 0
            self.r_const_i = 0

        self.r_const += (val << self.r_const_i)
        self.r_const_i += 1

    def nextSymbol(self):
        px,py,_,_ = self.sym_list[self.r_pointer]

        next_sym = None
        dist_sym = None
        cost_sym = None

        # TODO: improve distance checking algorithm
        for i in range(len(self.sym_list)):
            if (i != self.r_prev_pointer) and (i != self.r_pointer):

                x,y,_,_ = self.sym_list[i]
                d = dist2(x,y,px,py)
                diff = abs(bound_ang(ang(x,y,px,py) - self.r_pointer_dir))

                cost = d * diff

                # angle checking
                if (self.r_force_launch and diff > (np.pi / 3)):
                    continue

                if (diff > (np.pi / 2)):
                    continue

                if ((d < dist_sym or dist_sym == None) and d < pow(120,2)):
                    dist_sym = d
                    next_sym = i

                if (dist_sym == None and (cost < cost_sym or cost_sym == None)):
                    cost_sym = cost
                    next_sym = i


        if next_sym == None:
            self.err(22, 'you\'re lost', [self.r_pointer], theta=self.r_pointer_dir)

        return next_sym

### CLI

def help(cmd):
    print('Usage: ' + cmd + ' -i image [ -d ] [ -q ] [ -h ]')
    print('\nOptions:')
    print('\t-d : run in debug mode')
    print('\t-q : run in quiet mode (only print program output)')
    print('\t-h : print this help message')

def run(argv):
    try:
        opts, args = getopt.getopt(argv[1:],"hi:a:dq")
    except getopt.GetoptError:
        help(argv[0])
        sys.exit(1)

    image_path = None
    debug = False
    quiet = False

    for opt, arg in opts:
        if opt == '-h':
            help(argv[0])
            sys.exit()
        elif opt == '-i':
            image_path = arg
        elif opt == '-d':
            debug = True
        elif opt == '-q':
            quiet = True

    if image_path == None:
        help(argv[0])
        sys.exit(-1)

    p = Program(image_path, debug=debug, quiet=quiet)
    p.run()

if __name__ == '__main__':
    run(sys.argv[:])
