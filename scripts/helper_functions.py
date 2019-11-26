#!/usr/bin/env python

import cv2
import math
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from itertools import combinations

class text_colors:
  '''
  This function is used to produce colored output when using the print command.
  e.g.
  print text_colors.WARNING + "Warning: Something went wrong." + text_colors.ENDCOLOR
  print text_colors.FAIL + "FAILURE: Something went wrong." + text_colors.ENDCOLOR
  print text_colors.BOLD + "Something important in bold." + text_colors.ENDCOLOR
  '''
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDCOLOR = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

# Image Processing Functions
def convert_image(msg, flag = False):
  '''
  Converts ROS image to OpenCV image, then blocks out errant pixels and rectifies.
  '''
  bridge = CvBridge()
  try:
    # cv_bridge automatically scales, we need to remove that behavior
    if bridge.encoding_to_dtype_with_channels(msg.encoding)[0] in ['uint16', 'int16']:
      mono16 = bridge.imgmsg_to_cv2(msg, '16UC1')
      mono8 = numpy.array(numpy.clip(mono16, 0, 255), dtype=numpy.uint8)
      img = mono8
    elif 'FC1' in msg.encoding:
      # floating point image handling
      img = bridge.imgmsg_to_cv2(msg, "passthrough")
      _, max_val, _, _ = cv2.minMaxLoc(img)
      if max_val > 0:
        scale = 255.0 / max_val
        img = (img * scale).astype(numpy.uint8)
      else:
        img = img.astype(numpy.uint8)
    else:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
  except CvBridgeError as e:
    print text_colors.WARNING + 'Warning: Image converted unsuccessfully before processing.' + text_colors.ENDCOLOR
    raise e

  # Black out a rectangle in the top left of the image since there are often erroneous pixels there (image conversion error perhaps?)
  cv2.rectangle(img, (0,0), (30,1), 0, cv2.FILLED) # cv2.cv.CV_FILLED)
  show_image('blacked out', img, flag)

  return img

def show_image(title, img, flag = True, duration = 1):
  '''
  Uses cv2.imshow() to display an image to the screen.

  Arguments:
    flag  If flag is false, image will not be shown. Useful when there are multiple show_image calls throughout a script that are flagged with a single boolean.
    duration  Number of ms to display the image. A value of 0 displays the image until a key is pressed.
  '''
  if flag:
    cv2.imshow(title,img)
    cv2.waitKey(duration)

def draw_axes(img, corners, imgpts):
    img_deep_copy = np.array(img)
    corner = tuple(corners[0].ravel())
    cv2.line(img_deep_copy,corner,tuple(imgpts[2].ravel()),255,3)
    cv2.line(img_deep_copy,corner,tuple(imgpts[1].ravel()),170,3)
    cv2.line(img_deep_copy,corner,tuple(imgpts[0].ravel()),85,3)
    return img_deep_copy

def rectify(img, mtx, dist):
  # Undistortion
  h,w = img.shape[:2]
  mtx_new,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  return cv2.undistort(img,mtx,dist,None,mtx_new)

class CentroidFinder(object):
  def __init__(self, show_images, flag_debug, binary_threshold):
    self._img = None
    self._show_images = show_images
    self._debug = flag_debug
    self._binary_threshold = binary_threshold

  def get_centroids(self, img):
    '''
    Returns a tuple of tuples containing all centroids found in the image. Return tuple can be arbitrary length.

    Arguments:
      img   An OpenCV image

    Example return:
      ((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6),(x7,y7),(x8,y8),(x9,y9))

    IMAGE MUST BE RECTIFIED BEFORE THIS STEP
    '''

    # Set image
    self._img = np.array(img)

    # Find centroids and reshape for ROS publishing
    centroids = self._obtain_initial_centroids()

    for c in centroids:
      center = (int(c[0]),int(c[1]))
      cv2.circle(self._img, center, 3, 255, 3)

    # Return centroids and image with centroids highlighted
    return centroids, self._img

  def _obtain_initial_centroids(self):
    '''
    This function processes the image and returns the centroid of all contours. Centroids are in image coordinates.

    IMAGE MUST BE RECTIFIED BEFORE THIS STEP
    '''

    # Binarize
    max_value = 255
    block_size = 5
    const = 1
    threshold_value = self._binary_threshold
    _,self._img = cv2.threshold(self._img,threshold_value,max_value,cv2.THRESH_BINARY)

    if self._show_images:
      cv2.imshow('binary',self._img)
      cv2.waitKey(1)

    # Morph image to 'close' the shapes that are found
    kernel = np.ones((2,2),np.uint8)
    self._img = cv2.dilate(self._img,kernel,iterations = 1)
    self._img = cv2.erode(self._img,kernel,iterations = 1)

    if self._show_images:
      cv2.imshow('morph',self._img)
      cv2.waitKey(1)

    # Find contours
    # contours, _ = cv2.findContours(self._img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(self._img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    if self._show_images:
      img_temp = np.array(self._img)
      cv2.drawContours(img_temp, contours, -1, (255,255,255), 3)
      cv2.imshow('contours',img_temp)
      cv2.waitKey(1)

    # Extract centroids
    centroids = []
    all_centers = set([])
    max_y,max_x = self._img.shape[:2]
    edge_of_img_border = 5
    near_buff = 5
    for c in contours:
      # Obtain the coordinates of the center of the contour
      (x,y),_ = cv2.minEnclosingCircle(c)
      center = (int(x),int(y))
      # If we have already detected a point at (or very near) a given coordinate, do not add it again.
      if center not in all_centers:
        all_centers.add(center)
        # Add nearby buffer so that we do not get redundant centroids
        for i in range(-near_buff,near_buff + 1):
          for j in range(-near_buff,near_buff + 1):
            nearby_center = (center[0] + i,center[1] + j)
            all_centers.add(nearby_center)
        # Add the centroid to the collective list. Region of interest is added here since a truncated image will be offset by the amount of truncation.
        centroids.append((x, y))

    # Return the centroids found
    return tuple(centroids)

class NoiseFilter(object):
  def __init__(self, show_images, flag_debug, rotate):
    self._show_images = show_images
    self._img = None
    self._debug = flag_debug
    self._rotate = rotate

  def filter_noise(self, img, centroids):
    '''
    This function takes in a tuple of tuples containing centroids. It filters these centroids for noise and returns the result. The feature is expected to have eight points arranged in two parallel rows of four equally spaced points.

    Arguments:
      img   An OpenCV image
      centroids   A tuple of arbitrary length where each element is a tuple of length two
    '''
    self._img = np.array(img)

    if len(centroids) < 8:
      if self._debug:
        print text_colors.FAIL + "Failure: Too few centroids after initial selection." + text_colors.ENDCOLOR
      return (), self._img

    # We expect 8 feature points. Filter to remove any extras.
    if len(centroids) > 8:

      if self._debug:
        print text_colors.OKBLUE + "Note: Zeroeth filter applied." + text_colors.ENDCOLOR
      centroids = self._zeroeth_round_centroid_filter(centroids)

      if self._debug:
        print text_colors.OKBLUE + "Note: First filter applied." + text_colors.ENDCOLOR
      centroids = self._first_round_centroid_filter(centroids)

      # TODO: rewrite 2nd round filter to be camera orientation independent (i.e. if camera is mounted at 90 degrees, make 2nd round filter work still)
      # if len(centroids) > 8:
      #   if self._debug:
      #     print text_colors.OKBLUE + "Note: Second filter applied." + text_colors.ENDCOLOR
      #   centroids = self._second_round_centroid_filter(centroids)

    # The filters may have wiped out too many points. Check if that's the case.
    if len(centroids) < 8:
      if self._debug:
        print text_colors.FAIL + "Failure: Too few centroids after filtering." + text_colors.ENDCOLOR
      return (), self._img

    for c in centroids:
      center = (int(c[0]),int(c[1]))
      cv2.circle(self._img, center, 3, 255, 3)

    return centroids, self._img

  def _dist(self, coords1,coords2):
    '''
    Return the euclidean distance between two points with units as passed.
    '''
    a = coords1[0] - coords2[0]
    b = coords1[1] - coords2[1]
    distance = np.sqrt(a**2 + b**2)
    return distance

  def _cluster(self, data, maxgap):
    '''
    This method is taken from Raymond Hettinger at http://stackoverflow.com/questions/14783947/grouping-clustering-numbers-in-python

    Arrange data into groups where successive elements
    differ by no more than *maxgap*

    >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
    [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

    >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
    [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]
    '''

    data = np.array(data)
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
      if np.absolute(x - groups[-1][-1]) <= maxgap:
        groups[-1].append(x)
      else:
        groups.append([x])
    return groups

  def _zeroeth_round_centroid_filter(self,centroids):
    '''
    Takes subsets of four and returns all centroids that have low residuals after a linear fit.
    '''

    orig_centroids = centroids

    # Find rows based upon linear fit residuals
    subsets = combinations(centroids,4)
    rows = []
    for s in subsets:
      x = [p[0] for p in s]
      y = [p[1] for p in s]
      if self._rotate:
        _, residuals, _, _, _ = np.polyfit(y, x, 1, full = True)
      else:
        _, residuals, _, _, _ = np.polyfit(x, y, 1, full = True)
      if residuals < 4:
        rows.append(list(s))

    # Get all centroids that exist in a low-residual subset
    set_of_centroids = set([r[i] for r in rows for i in range(0,len(r))])
    centroids = list(set_of_centroids)

    # If zeroeth filter fails, return the original list
    if len(centroids) < 8:
      return orig_centroids

    return centroids

  def _first_round_centroid_filter(self, centroids):
    '''
    This function takes in a tuple of tuples containing centroids and returns a similar tuple of tuples that has been filtered to remove suspected erroneous values. Filtering is performed by clustering x and y image coordinates, and then intersecting those clusters.
    '''

    def one_sub_cluster_of_at_least_8(clstr):
      ret_val = False
      for c in clstr:
        if len(c) >= 8:
          ret_val = True
      return ret_val

    # First, create two clusters of the centroids, one which is clustered by x and one by y.
    x_coords = [c[0] for c in centroids]
    y_coords = [c[1] for c in centroids]

    std_dev_x = np.std(x_coords)
    std_dev_y = np.std(y_coords)

    k = 0.1
    k_step = 0.01
    x_clstr = self._cluster(x_coords,std_dev_x*k)
    y_clstr = self._cluster(y_coords,std_dev_y*k)

    x_iter = 1
    while not one_sub_cluster_of_at_least_8(x_clstr):
      x_clstr = self._cluster(x_coords,std_dev_x*k)
      k = k + k_step
      x_iter = x_iter + 1

    y_iter = 1
    while not one_sub_cluster_of_at_least_8(y_clstr):
      y_clstr = self._cluster(y_coords,std_dev_y*k)
      k = k + k_step
      y_iter = y_iter + 1

    # Since we specified that at least one of our clusters in both x and y have at least 8 points, we need to find the x and y cluster_of_interest that contains the 8 points. With a significant amount of noise, it is possible that we have more than one cluster with 8 points. Print a warning if this happens.
    x_cluster_of_interest = []
    y_cluster_of_interest = []
    for x in x_clstr:
      if len(x) >= 8:
        x_cluster_of_interest.append(x)
    for y in y_clstr:
      if len(y) >= 8:
        y_cluster_of_interest.append(y)

    if len(x_cluster_of_interest) > 1:
      print text_colors.WARNING + 'Warning: Too many x clusters of interest.' + text_colors.ENDCOLOR
      return []

    if len(y_cluster_of_interest) > 1:
      print text_colors.WARNING + 'Warning: Too many y clusters of interest.' + text_colors.ENDCOLOR
      return []

    x_cluster_of_interest = x_cluster_of_interest[0]
    y_cluster_of_interest = y_cluster_of_interest[0]

    # Gather centroids of interest from clusters of interest
    x_centroids_of_interest = set([])
    for x_val in x_cluster_of_interest:
      indices = [i for i,x in enumerate(x_coords) if x == x_val]
      for index in indices:
        y_val = y_coords[index]
        centroid_to_add = (x_val,y_val)
        x_centroids_of_interest.add(centroid_to_add)
    y_centroids_of_interest = set([])
    for y_val in y_cluster_of_interest:
      indices = [i for i,y in enumerate(y_coords) if y == y_val]
      for index in indices:
        x_val = x_coords[index]
        centroid_to_add = (x_val,y_val)
        y_centroids_of_interest.add(centroid_to_add)

    # Now that we have clustered by x and y centroids, we need to take the intersection of these clusters, as this is likely our feature.
    centroids_of_interest = [c for c in y_centroids_of_interest if c in x_centroids_of_interest]

    # Attempt to recover grouping in the case where there are not 8 centroids shared by the x and y clusters.
    if len(centroids_of_interest) < 8 and len(centroids_of_interest) > 0:
      # Here we have the case where the number of centroids shared by x and y clusters is less than the number of feature points. This means that the x and y clusters are being thrown off by false positive measurements. At this point, we must decide which 8 points are the correct centroids.

      # First, we take the points that are shared by the x and y clusters. We calculate the average position of these points and use distance to this average position as a metric for choosing the remaining correct centroids.
      avg_pos_x = 0
      avg_pos_y = 0
      for c in centroids_of_interest:
        avg_pos_x = avg_pos_x + c[0]
        avg_pos_y = avg_pos_y + c[1]
      avg_pos_x = avg_pos_x/len(centroids_of_interest)
      avg_pos_y = avg_pos_y/len(centroids_of_interest)
      avg_pos = (avg_pos_x,avg_pos_y)

      dist_to_avg_pos = []
      for c in centroids_of_interest:
        dist_to_avg_pos.append(self._dist(c,avg_pos))

      dist_to_avg_pos_mean = np.mean(dist_to_avg_pos)
      dist_to_avg_pos_std = np.std(dist_to_avg_pos)

      # Now that we have the average position of the accepted centroids, we must query the remaining centroids for those that are nearest to this average position.
      remaining_centroids = [c for c in centroids if c not in centroids_of_interest]

      selected_centroids = []
      for c in remaining_centroids:
        if np.absolute(self._dist(c,avg_pos) - dist_to_avg_pos_mean) < 5*dist_to_avg_pos_std:
          selected_centroids.append(c)

      for c in selected_centroids:
        centroids_of_interest.append(c)

    centroids_of_interest = tuple([tuple(c) for c in centroids_of_interest])
    return centroids_of_interest

  def _second_round_centroid_filter(self, centroids):
    '''
    This function takes in a list of centroids and returns a similar list that has been filtered to remove suspected erroneous values. Filtering is performed by clustering according to pairwise slope value.
    '''

    def at_least_one_cluster_of_at_least_12(clstr):
      for c in clstr:
        if len(c) >= 12:
          return True
      return False

    # Create a list of all subsets of 2 points
    subsets = combinations(centroids,2)

    # Calculate the slope of each pair
    slopes_and_points = []
    slopes = []
    for s in subsets:
      # Set point 1 to be the leftmost point and point 0 to be the right most point
      if s[0][0] < s[1][0]:
        pt0 = s[0]
        pt1 = s[1]
      else:
        pt0 = s[1]
        pt1 = s[0]
      # Determine the slope of the line. Handle special cases of slopes.
      rise = pt1[1]-pt0[1]
      run = pt1[0]-pt0[0]
      if run == 0 and rise == 0:
        # Do nothing. We are using the same point twice for some reason
        pass
      elif run == 0:
        # Do nothing. This is a vertical line and therefore is a pair of points with the points on different rows.
        pass
      else:
        # Store the slope and points together
        m = rise/run
        slopes_and_points.append((m,pt0,pt1))
        slopes.append(m)

    # Search the slopes_and_points list for point combinations which have nearly the same slope
    k = 0.005
    k_step = 0.005
    clustered_slopes = self._cluster(slopes,k)

    while not at_least_one_cluster_of_at_least_12(clustered_slopes) and k < 100*k_step:
      k = k + k_step
      clustered_slopes = self._cluster(slopes,k)

    slopes_of_interest = None
    for c in clustered_slopes:
      if len(c) >= 12:
        slopes_of_interest = c

    # Report an error if we do not detect a slope cluster with at least 12 points
    if slopes_of_interest is None:
      print text_colors.WARNING + 'Warning: Invalid slope clusters.' + text_colors.ENDCOLOR
      return []

    # Now that we have clustered by slope value, remove all subsets whose slope is not in the cluster
    for i in range(len(slopes_and_points)-1,-1,-1):
      if slopes_and_points[i][0] not in slopes_of_interest:
        del slopes_and_points[i]

    # Create a set of all of the points in our slope cluster so we have no duplicates.
    points = set([])
    for i in range(0,len(slopes_and_points)):
      pt0 = tuple(slopes_and_points[i][1])
      pt1 = tuple(slopes_and_points[i][2])
      points.add(pt0)
      points.add(pt1)

    return tuple(points)

class PnPSolver(object):
  def __init__(self, mtx, dist, show_images, flag_debug, rotate):
    self._show_images = show_images
    self._debug = flag_debug
    self._mtx = mtx
    self._dist = dist
    self._print_feature_name = True
    self._img = None
    self._rvecs = None
    self._tvecs = None
    self._rotate = rotate

  def solve_pnp(self, img, centroids):
    '''
    Corresponds each image point in centroids to a known real-world coordinate, then solves the Perspective-n-Point problem for that assignment.
    '''

    # Set image
    self._img = np.array(img)

    # Assign points
    assigned_rows = self._assign_points(centroids)
    total_points_assigned = len(assigned_rows[0]) + len(assigned_rows[1])
    if not total_points_assigned == 8:
      if self._debug:
        print text_colors.FAIL + "Failure: Failure to assign points correctly." + text_colors.ENDCOLOR
      return (None,None,None), (None,None,None), (None,None,None), self._img

    # Extract pose
    position, yawpitchroll, orientation = self._pose_extraction(assigned_rows)

    return position, yawpitchroll, orientation, self._img

  def _assign_points(self, centroids):
    '''
    This function assigns image points (centroids) to their corresponding real-world coordinate based upon a linear regression of subsets of four points. The feature has two rows of 4 linear points, so we expect two subsets of four points with relatively low residuals.
    '''

    centroids = [list(c) for c in centroids]

    if len(centroids) >= 8:
      # Find rows based upon linear fit residuals
      subsets = combinations(centroids,4)
      residual_list = []
      for s in subsets:
        x = [p[0] for p in s]
        y = [p[1] for p in s]
        if self._rotate:
          _, residuals, _, _, _ = np.polyfit(y, x, 1, full = True)
        else:
          _, residuals, _, _, _ = np.polyfit(x, y, 1, full = True)
        residual_list.append((residuals[0],list(s)))

      # Take the two subsets with lowest residual
      residual_list.sort(key=lambda x: x[0])
      rows = [residual_list[0][1], residual_list[1][1]]

      # Now we have both rows, so we must decide which is the top row and which is the bottom row. First, sort each row so that the points in each row are organized from right to left (if rotated, top to bottom) in the image.
      if self._rotate:
        for r in rows:
          r.sort(key=lambda x: -x[1])
      else:
        for r in rows:
          r.sort(key=lambda x: x[0])

      # Then, use the first element of each row to determine which row is on top (if rotated determine which row is on the left)
      if self._rotate:
        if rows[0][0][0] < rows[1][0][0]:
          top_row    = rows[0]
          bottom_row = rows[1]
        else:
          top_row    = rows[1]
          bottom_row = rows[0]
      else:
        if rows[0][0][1] < rows[1][0][1]:
          top_row    = rows[0]
          bottom_row = rows[1]
        else:
          top_row    = rows[1]
          bottom_row = rows[0]

      top_row = tuple([tuple(c) for c in top_row])
      bottom_row = tuple([tuple(c) for c in bottom_row])

      # Draw top and bottom rows in order
      k = 0
      for i in range(0,len(bottom_row)):
        k = k + 1
        c = bottom_row[i]
        center = (int(c[0]),int(c[1]))
        cv2.circle(self._img, center, 3, 31.875*k, 5)
      for i in range(0,len(top_row)):
        k = k + 1
        c = top_row[i]
        center = (int(c[0]),int(c[1]))
        cv2.circle(self._img, center, 3, 31.875*k, 5)

      return (bottom_row,top_row)
    else:
      return ((),())

  def _pose_extraction(self, centroids):
    '''
    Solves the Perspective-n-Point problem for the given centroids. The passed in centroids are assumed to have the following format:
    (bottom_row,top_row)
      where
        bottom_row = ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
        top_row = ((x5,y5),(x6,y6),(x7,y7),(x8,y8))

    From the UAV's perspective, the points are numbered from left to right starting with the bottom row.

    Arguments
      centroids   A tuple of tuples of tuples containing the image coordinates of the top and bottom rows of the feature.
    '''

    bottom_row = centroids[0]
    top_row = centroids[1]

    # Calculate pose. First, define object points. The units used here, [cm], will determine the units of the output. These are the relative positions of the beacons in NED GCS-frame coordinates (aft, port, down).
    objp = np.zeros((8,1,3), np.float32)

    # # VICON FEATURE
    if self._print_feature_name:
      print text_colors.OKGREEN + "Using vicon feature" + text_colors.ENDCOLOR
      self._print_feature_name = False
    row_aft = [0,-0.802] # [m]
    row_port = [[0.0, -0.161, -0.318, -0.476],[0.0, -0.159, -0.318, -0.472]] # [m]
    row_down = [0,-0.256] # [m]
    # Lower row of beacons
    objp[0] = [ row_aft[0], row_port[0][0], row_down[0]]
    objp[1] = [ row_aft[0], row_port[0][1], row_down[0]]
    objp[2] = [ row_aft[0], row_port[0][2], row_down[0]]
    objp[3] = [ row_aft[0], row_port[0][3], row_down[0]]
    # Upper row of beacons
    objp[4] = [ row_aft[1], row_port[1][0], row_down[1]]
    objp[5] = [ row_aft[1], row_port[1][1], row_down[1]]
    objp[6] = [ row_aft[1], row_port[1][2], row_down[1]]
    objp[7] = [ row_aft[1], row_port[1][3], row_down[1]]

    # # ORIENTATION TEST VICON FEATURE
    # if self._print_feature_name:
    #   print text_colors.OKGREEN + "Using orientation test vicon feature" + text_colors.ENDCOLOR
    #   self._print_feature_name = False
    # row_x = [0, -0.802]
    # row_y = [[0.0, 0.161, 0.318, 0.476],[0.0, 0.159, 0.318, 0.472]]
    # row_z = [0,0.256]
    # # Lower row of beacons
    # objp[0] = [ row_x[0], row_y[0][0], row_z[0]]
    # objp[1] = [ row_x[0], row_y[0][1], row_z[0]]
    # objp[2] = [ row_x[0], row_y[0][2], row_z[0]]
    # objp[3] = [ row_x[0], row_y[0][3], row_z[0]]
    # # Upper row of beacons
    # objp[4] = [ row_x[1], row_y[1][0], row_z[1]]
    # objp[5] = [ row_x[1], row_y[1][1], row_z[1]]
    # objp[6] = [ row_x[1], row_y[1][2], row_z[1]]
    # objp[7] = [ row_x[1], row_y[1][3], row_z[1]]

    # # TRAILER FEATURE
    # if self._print_feature_name:
    #   print text_colors.OKGREEN + "Using trailer feature" + text_colors.ENDCOLOR
    #   self._print_feature_name = False
    # row_aft = [0,-1.397] # [m]
    # row_port_lower = [-0.297, -0.895, -1.495, -2.099] # [m]
    # row_port_lower = [elm - (-0.297) for elm in row_port_lower]
    # row_port_upper = [-0.300, -0.899, -1.498, -2.099] # [m]
    # row_port_upper = [elm - (-0.297) for elm in row_port_upper]
    # row_port = [row_port_lower, row_port_upper]
    # row_down = [0,-1.22] # [m]
    # # Lower row of beacons
    # objp[0] = [ row_aft[0], row_port[0][0], row_down[0]]
    # objp[1] = [ row_aft[0], row_port[0][1], row_down[0]]
    # objp[2] = [ row_aft[0], row_port[0][2], row_down[0]]
    # objp[3] = [ row_aft[0], row_port[0][3], row_down[0]]
    # # Upper row of beacons
    # objp[4] = [ row_aft[1], row_port[1][0], row_down[1]]
    # objp[5] = [ row_aft[1], row_port[1][1], row_down[1]]
    # objp[6] = [ row_aft[1], row_port[1][2], row_down[1]]
    # objp[7] = [ row_aft[1], row_port[1][3], row_down[1]]

    # Define feature points by the correspondences determined above. The bottom row corresponds to the lower row of beacons, and the top row corresponds to the upper row. Each row in is arranged with the leftmost point in the image in the first index, and so on.
    feature_points = np.zeros((8,1,2), np.float32)
    # Lowermost Subset
    feature_points[0] = bottom_row[0]
    feature_points[1] = bottom_row[1]
    feature_points[2] = bottom_row[2]
    feature_points[3] = bottom_row[3]
    # Uppermost Subset
    feature_points[4] = top_row[0]
    feature_points[5] = top_row[1]
    feature_points[6] = top_row[2]
    feature_points[7] = top_row[3]

    # Find rotation and translation vectors
    use_prev_solution = False
    if self._rvecs is None:
      use_prev_solution = False
    # flag_success,rvecs,tvecs = cv2.solvePnP(objp,feature_points,self._mtx,self._dist,self._rvecs,self._tvecs,use_prev_solution,cv2.CV_ITERATIVE)
    flag_success,rvecs,tvecs = cv2.solvePnP(objp,feature_points,self._mtx,self._dist,self._rvecs,self._tvecs,use_prev_solution,cv2.SOLVEPNP_ITERATIVE)

    if flag_success:
      self._rvecs = rvecs
      self._tvecs = tvecs

      # Calculate pose
      Pc = tuple(feature_points[0].ravel())
      Pc = np.array([[Pc[0]], [Pc[1]], [1]])
      Kinv = np.matrix(np.linalg.inv(self._mtx))
      R,_ = cv2.Rodrigues(self._rvecs)
      Rinv = np.matrix(np.linalg.inv(R))
      T = np.array(self._tvecs)

      # Calculate relative position
      position = Rinv*(Kinv*Pc-T)
      position = [float(val) for val in position]

      # Get relative orientation as yaw, pitch, roll with respect to feature
      orientation = self._orientation_decomposition(R)
      yawpitchroll = self._zyx2ypr(orientation)

      # Draw axes on image
      axis_len = 0.6 # [m]
      axis = np.float32([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1,3)
      imgpts,_ = cv2.projectPoints(axis,rvecs,tvecs,self._mtx,self._dist)
      self._img = draw_axes(self._img,feature_points,imgpts)

      if self._debug:
        print text_colors.OKGREEN + "Success." + text_colors.ENDCOLOR

    else:
      self._rvecs = None
      self._tvecs = None
      position = (None, None, None)
      orientation = (None, None, None)
      yawpitchroll = (None, None, None)

    # Return the obtained pose, rvecs, and tvecs
    return position, yawpitchroll, orientation

  def _orientation_decomposition(self,R):
    '''
    Returns rotation in degrees of the camera w.r.t. GCS.
    '''
    sin_y = math.sqrt(R[2,0]*R[2,0] + R[2,1]*R[2,1])
    singular = sin_y < 1e-6
    if not singular:
      z = math.atan2(R[2,0], R[2,1])
      y = math.atan2(sin_y, R[2,2])
      x = math.atan2(R[0,2], -R[1,2])
    else: # Gimbal lock
      z = 0
      y = math.atan2(sin_y, R[2,2])
      x = 0
    z = z*180/np.pi
    y = y*180/np.pi
    x = x*180/np.pi

    # Unsure why these are needed, but it works
    z = (-90 - z) % 360
    y = y + 90

    if self._rotate:
      x = x + 90

    # Limit z to +- 180
    if z > 180:
      z = z - 360

    return [z,y,x]

  def _zyx2ypr(self, zyx_angles):
    '''
    Converts rotation angles in degrees to relative yaw, pitch, and roll in degrees.
    '''
    z = zyx_angles[0]
    y = zyx_angles[1]
    x = zyx_angles[2]

    yaw = z
    pitch = y - 180
    # if self._rotate:
    #   x = x + 90
    if x > 0:
      roll = -(x - 180)
    else:
      roll = -(x + 180)

    return [yaw,pitch,roll]
