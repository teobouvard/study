# mst.py
# Author: Ronald L. Rivest
# Date: April 7, 2006
# Illustrates computations of minimum spanning tree;
# computes mst for set of bouncing balls.
# Uses PyGame for graphics
# See book Introduction to Algorithms by Cormen, Leiserson, Rivest, Stein
# (MIT Press;McGraw Hill) Chapter 23, for discussion of minimum spanning trees.

###########################################################################
### License stuff                                                       ###
###########################################################################
"""
Copyright (C) 2006  Ronald L. Rivest

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.

[Note: the pygame library, upon which this is based, comes with the
 Lesser GPL license (LGPL).]

Email: rivest@mit.edu
Mail:  Room 32-G692, MIT, Cambridge, MA 02139
"""
###########################################################################
###########################################################################

import math
import random
import time
import sys

# globals

balls = []
number_balls = 30                 # starting value                   
ball_min_radius = 8.0            # pixels
ball_max_radius = 32.0
speed = 3.0                      # pixels / click
balls_paused = False   
mst_edge_list = []

###########################################################################
### Display (pygame) related stuff                                      ###
###########################################################################

import pygame
from pygame.locals import *

size_x = 1200                    # screen size dummy values, set to fullscreen size later
size_y = 800

color_scheme = 1                 # 0 = white background, 1 = black background
background_color = [0,0,0]
line_color = [250,250,250]

line_width = 4

show_balls = True
show_edges = True

show_edges_kruskal = False       # display edges one by one, in kruskal order
show_edges_prim = False          # display edges one by one, in prim order
show_edge_pause = 1.0            # how many seconds to pause as each edge is displayed


def set_color_scheme(scheme):
	""" 
	Set color scheme. 
	scheme = 0 (white background) or 1 (black background)
	"""
	global line_color, background_color
	if scheme == 0:
		line_color = [0,0,0]
		background_color = [250,250,250]
	if scheme == 1:
		line_color = [250,250,250]
		background_color = [0,0,0]

class Ball:
	""" Implements a point/ball/vertex """

	def __init__(self):
		self.x = random.uniform(0.0,size_x)
		self.y = random.uniform(0.0,size_y)
		angle = random.uniform(0.0,math.pi)
		ball_speed = random.uniform(0.0,2.0)
		self.vx = math.sin(angle) * ball_speed
		self.vy = math.cos(angle) * ball_speed
		self.d = 0.0              # distance to tree in prim
		self.pred = None          # predecessor ball in prim
		self.color = [120+int(random.random()*130),
					  120+int(random.random()*130),
					  120+int(random.random()*130)]
		self.radius = int(random.uniform(ball_min_radius,ball_max_radius))
		self.mass = float(self.radius**2)

	def draw(self,surface):
		""" Draw ball. """
		# colored portion
		pygame.draw.circle(surface,
						   self.color,
						   [int(self.x),int(self.y)],
						   self.radius,
						   0)  # width (0 means fill circle)
		# circumference line
		pygame.draw.circle(surface,
						   line_color,
						   [int(self.x),int(self.y)],
						   self.radius,
						   1)  # width

def initialize_screen():
	""" Set up display screen. """
	global screen, background, size_x, size_y
	pygame.init()

	# get size of fullscreen display into size_x, size_y
	modes = pygame.display.list_modes()    # defaults to fullscreen
	modes.sort()                           # largest goes last
	size_x,size_y = modes[-1]

	screen = pygame.display.set_mode((size_x, size_y),
									 pygame.FULLSCREEN )
	# following line is irrelevant for full-screen display
	pygame.display.set_caption('Minimum Spanning Tree program')

def initialize_font():
	global font
	font = pygame.font.Font(None, 36)

def initialize_background(color):
	""" Set background to background_color. """
	global background, screen

	background = pygame.Surface(screen.get_size())
	background = background.convert()
	background.fill(color)

def show_background():
	""" Write background onto screen and show it. """
	global background, screen

	screen.blit(background, (0, 0))
	pygame.display.flip()

def show_text_screen(msgs):
	""" 
	Show a screen of text.
	Return True if user wishes to quit out of this text screen.
	"""
	global font, background
	initialize_background([250,250,250])

	y = 100
	for msg in msgs:
		text = font.render(msg, 
						   1,             # antialias
						   (10, 10, 10))  # color
		textpos = text.get_rect()
		textpos.centerx = background.get_rect().centerx
		textpos.centery = y
		y += 40
		background.blit(text, textpos)

	show_background()
	while 1:
		for event in pygame.event.get():
			if event.type == QUIT:
				return True
			elif event.type == KEYDOWN and event.key == K_ESCAPE:
				return True
			elif event.type == KEYDOWN and event.key == K_SPACE:
				return False
			elif event.type == KEYDOWN and event.key == K_F1:
				return show_help_screen()
			elif event.type == KEYDOWN and event.key == K_F2:
				return show_info_screen()

def show_welcome_screen():
	""" 
	Show initial welcome / help screen. 
	"""

	msgs = ["MST -- Minimum Spanning Tree Demo Program",
			"An experiment in pedagogic eye candy",
			"(c) Ronald L. Rivest. 4/7/2006. Version 1.0.  GPL License.",
			" ",
			"You'll see colored bouncing balls, and edges connecting them.",
			" ",
			"The MST edges connect all the balls with minimum total edge length.",
			" ",
			"SPACE proceeds",
			"F1 shows a help screen",
			"F2 shows an info screen",
			"ESC quits",
			]
	return show_text_screen(msgs)

def show_help_screen():
	""" 
	Show help screen. 
	Return True if user wishes to quit out of help screen.
	"""

	msgs = ["Up arrow increases number of balls",
			"Down arrow decreases number of balls",
			" ",
			"Right arrow increases their speed",
			"Left arrow decreases their speed",
			"SPACE pauses/restarts ball motion",
			" ",
			"b turns ball display on/off",
			"e turns edge display on/off",
			"c flips background color (black/white)",
			" ",
			"k turns on incrementally showing how edges selected with kruskal",
			"p turns on incrementally showing how edges selected with prim",
			"n turns off incremental edge display (no edge display delays)",
			" ",
			"SPACE proceeds",
			"F1 shows this help screen",
			"F2 shows info screen",
			"ESC quits"
			]
	return show_text_screen(msgs)

def show_info_screen():
	""" 
	Show info screen. 
	Return True if user wishes to quit out of info screen.
	"""

	msgs = ["See Cormen/Leiserson/Rivest/Stein 'Introduction to Algorithms'",
			"(MIT Press/McGraw-Hill) for more information on MST algorithms.",
			" ",
			"Prim's algorithm starts at one ball, and incrementally attaches one ball",
			"after another to growing MST, in cheapest way possible at each step.",
			" ",
			"Kruskal's algorithm considers edges in order of increasing length, and",
			"includes an edge in MST if it connects two previously unconnected trees.",
			" ",
			"Source code at http://theory.csail.mit.edu/~rivest/mst.py",
			"Executable at http://theory.csail.mit.edu/~rivest/mst.exe",
			" ",
			"SPACE proceeds",
			"F1 shows help screen",
			"F2 shows this info screen",
			"ESC quits"
			]
	return show_text_screen(msgs)

def display_edges(edge_list):
	""" Display list of edges. """
	global show_edges_kruskal, show_edge_pause, show_edges
	if show_edges_kruskal:
		L = [(dist(b0,b1),b0,b1) for b0,b1 in edge_list]
		L.sort()
		edge_list = [(b0,b1) for d,b0,b1 in L]
	if show_edges_prim or show_edges_kruskal:
		show_background()
		time.sleep(show_edge_pause)
	for b0,b1 in edge_list:
		if show_edges:
			pygame.draw.line(background,
							 line_color,
							 (b0.x,b0.y),  # start
							 (b1.x,b1.y),  # end
							 line_width)   # width
			display_dot_at_ball_center(b0)
			display_dot_at_ball_center(b1)
			# to show lines slowly, as they are drawn:
			if show_edges_prim or show_edges_kruskal:
				show_background()
				time.sleep(show_edge_pause)
				if handle_user_input():
					sys.exit()

def display_dot_at_ball_center(b):
	""" Display a nice dot at center of ball b. """
	global background, line_color

	pygame.draw.circle(background,
					   line_color,
					   [int(b.x),int(b.y)],
					   7,
					   0)  # width (0 means fill circle)

def display_balls_and_edges(balls, edge_list):
	""" Show balls and edges. """
	global background, background_color, line_color
	global show_balls, show_edges
	background.fill(background_color)
	if show_balls:
		for b in balls:
			b.draw(background)
	if show_edges:
		if show_edges_prim and len(balls)>0:
			b = balls[0]
			display_dot_at_ball_center(b)
			time.sleep(show_edge_pause)
		display_edges(edge_list)
	show_background()

###########################################################################
### USER INPUT                                                          ###
###########################################################################

def handle_user_input():
	""" 
	Detect keypresses, etc., and handle them. 
	Return True iff user requests program to quit
	"""
	global number_balls, balls, speed, size_x, size_y
	global color_scheme, show_balls, show_edges, balls_paused
	global show_edges_prim, show_edges_kruskal
	for event in pygame.event.get():
		if event.type == QUIT:
			return True
		elif event.type == KEYDOWN and event.key == K_ESCAPE:
			return True
		elif event.type == KEYDOWN and event.key == K_F1:
			if show_help_screen():
				return True
		elif event.type == KEYDOWN and event.key == K_F2:
			if show_info_screen():
				return True
		elif event.type == KEYDOWN and event.key == K_DOWN:
			number_balls = max(0,
							   min(number_balls-1,
								   int(number_balls*0.80)))
			balls = balls[:number_balls]
		elif event.type == KEYDOWN and event.key == K_UP:
			number_balls = max(number_balls+1,
							   int(number_balls*1.20))
			while len(balls)<number_balls:
				balls.append(Ball())
		elif event.type == KEYDOWN and event.key == K_LEFT:
			speed /= 1.4
		elif event.type == KEYDOWN and event.key == K_RIGHT:
			speed *= 1.4
			speed = min(speed,size_x/3,size_y/3)
		elif event.type == KEYDOWN and event.key == K_b:
			show_balls = not show_balls
		elif event.type == KEYDOWN and event.key == K_e:
			show_edges = not show_edges
		elif event.type == KEYDOWN and event.key == K_c:
			color_scheme = (1+color_scheme)%2
			set_color_scheme(color_scheme)
		elif event.type == KEYDOWN and event.key == K_SPACE:
			balls_paused = not balls_paused
		elif event.type == KEYDOWN and event.key == K_p:
			show_edges_prim = True
			show_edges_kruskal = False
		elif event.type == KEYDOWN and event.key == K_k:
			show_edges_kruskal = True
			show_edges_prim = False
		elif event.type == KEYDOWN and event.key == K_n:
			show_edges_kruskal = False
			show_edges_prim = False
	return False

#################################################################################
### Routines related to computation of MST                                    ###
#################################################################################

def dist(b1,b2):
	""" Return distance between balls b1 and b2. """
	return (math.sqrt((b1.x-b2.x)**2 + (b1.y-b2.y)**2))

def compute_mst(balls):
	""" Compute MST of given set of balls. """
	# use prim's algorithm
	mst_edge_list = prim(balls)
	return mst_edge_list

def prim(balls):
	""" 
	Find mst of set of balls with Prim's algorithm.
	Return set of edges.
	"""

	if len(balls)==0:
		return []

	mst_edge_list = []

	b0 = balls[0]
	Q = balls[1:]
	for ball in Q: 
		ball.d = dist(ball,b0)
		ball.pred = b0

	while Q != []:
		min_d = 1e20
		for ball in Q:
			if ball.d < min_d:
				min_d = ball.d
				closest_ball = ball
		Q.remove(closest_ball)
		b0 = closest_ball
		b1 = closest_ball.pred
		mst_edge_list.append((b0,b1))
		for ball in Q:
			d = dist(ball,closest_ball)
			if d<ball.d:
				ball.d = d
				ball.pred = closest_ball

	return mst_edge_list

###########################################################################
### Routines related to ball motion and collision handling              ###
###########################################################################

def move_balls():
	""" Move all balls. """
	global balls
	for b in balls:
		move_ball(b)

def move_ball(b):
	""" 
	Move ball b one step, and bounce off walls.
	"""
	global speed, size_x, size_y

	b.x += b.vx * speed
	b.y += b.vy * speed

	r = b.radius

	left = 0.0
	if b.x < left + r:   # bounce off left wall
		b.x = (left + r)+(left+r-b.x)
		b.vx = -b.vx

	right = size_x
	if b.x > right - r:  # bounce off right wall
		b.x = (right - r)-(b.x-right+r)
		b.vx = -b.vx

	bottom = 0.0
	if b.y < bottom + r: # bounce off bottom wall
		b.y = (bottom + r)+(bottom+r-b.y)
		b.vy = -b.vy

	top = size_y
	if b.y > top - r:    # bounce off top wall
		b.y = top - r-(b.y-(top-r))
		b.vy = -b.vy

### Vector operations

def vadd(v1,v2):
	""" Return sum of vectors v1 and v2. """
	return [a+b for a,b in zip(v1,v2)]

def vsub(v1,v2):
	""" Return vector v1-v2 """
	return [a-b for a,b in zip(v1,v2)]

def vscale(s,v):
	""" Multiply vector v by the scalar s. """
	return [s*a for a in v]

def vlensq(v):
	""" Return the length squared of vector v. """
	return sum([x*x for x in v])

def vlen(v):
	""" Return the length of vector v. """
	return math.sqrt(vlensq(v))

def vdot(v1,v2):
	""" Return the dot product of vectors v1 and v2. """
	return sum([a*b for a,b in zip(v1,v2)])

def vunit(v):
	""" Return unit vector in same direction as v. """
	length = vlen(v)
	assert length > 0.0
	return vscale(1.0/length,v)

def handle_collisions(balls):
	""" 
	Detect and handle all ball-to-ball collisions.
	This uses an all-pairs approach, which is OK for a
	reasonable number of balls.
	"""
	for i in range(len(balls)):
		b0 = balls[i]
		for j in range(i):
			b1 = balls[j]
			d = dist(b0,b1)
			if d<=b0.radius+b1.radius:
				collide(b0,b1)

def collide(b1,b2):
	""" 
	Collide balls b1 and b2.

	Net result is that velocities of b1 and b2 may be changed.
	Detects "false collisions" where balls are close but actually
	moving away from each other; in this case it does nothing.
	"""

	# ball 1: mass, position, velocity
	m1 = b1.mass
	p1 = [b1.x,b1.y]
	v1 = [b1.vx,b1.vy]

	# ball 2: mass, position, velocity
	m2 = b2.mass
	p2 = [b2.x,b2.y]
	v2 = [b2.vx,b2.vy]

	# center of mass: position, velocity
	pc = vadd(vscale(m1/(m1+m2),p1),vscale(m2/(m1+m2),p2))
	vc = vadd(vscale(m1/(m1+m2),v1),vscale(m2/(m1+m2),v2))

	# return if at same position; can't do anything
	if p1 == p2: return

	u1 = vunit(vsub(p1,pc))      # unit vector towards m1 in cm coords
	w1 = vsub(v1,vc)             # velocity of m1 in cm coords
	z = vdot(w1,u1)              # amount of w1 in direction towards m1
	if z >= 0.0: return          # can't collide; m1 moving away from cm
	r1 = vscale(z,u1)            # velocity of m1 in cm coords along u1
	s1 = vsub(w1,vscale(2.0,r1)) # post-collision velocity in cm coords
	b1.vx, b1.vy = vadd(vc,s1)   # final velocity in global coords

	u2 = vunit(vsub(p2,pc))      # unit vector towards m2 in cm coords
	w2 = vsub(v2,vc)             # velocity of m2 in cm coords
	z = vdot(w2,u2)              # amount of w2 in direction towards m2
	if z >= 0.0: return          # can't collide; m2 moving away from cm
	r2 = vscale(z,u2)            # velocity of m2 in cm coords along u2
	s2 = vsub(w2,vscale(2.0,r2)) # post-collision velocity in cm coords
	b2.vx, b2.vy = vadd(vc,s2)   # final velocity in global coords


###########################################################################
### Main routine / event loop                                           ###
###########################################################################

def main():
	global number_balls, balls, balls_paused, background_color
	
	initialize_screen()
	initialize_background(background_color)
	pygame.key.set_repeat(500,300)   # for handling key repeats

	initialize_font()
	if show_welcome_screen():
		return
	
	# Make balls
	balls = [Ball() for i in range(number_balls)]

	# Event loop
	while 1:
		if handle_user_input(): return
		if not balls_paused:
			move_balls()
			handle_collisions(balls)
		mst_edge_list = compute_mst(balls)
		display_balls_and_edges(balls, mst_edge_list)

if __name__ == '__main__': main()
