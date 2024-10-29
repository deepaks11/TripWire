from shapely.geometry import Point, LineString, Polygon


class LineIntersectionTest:
    def __init__(self, detections, line_zones):
        self.detections = detections
        self.line = LineString([line_zones[0], line_zones[1]])

    def point_line_intersection_test(self):
        try:
            count = 0
            any_intersecting_or_touching_or_standing = False  # To track if any object is intersecting, touching, or standing on the line

            for xyxy in self.detections.xyxy:
                p_x1, p_y1, p_x2, p_y2 = xyxy.astype(int)
                person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]

                # Create a rectangle polygon for the person's bounding box
                rect_polygon = Polygon(person_coord)

                # Create line segments for the bounding box
                rect_lines = [
                    LineString([(p_x1, p_y1), (p_x1, p_y2)]),
                    LineString([(p_x1, p_y2), (p_x2, p_y2)]),
                    LineString([(p_x2, p_y2), (p_x2, p_y1)]),
                    LineString([(p_x2, p_y1), (p_x1, p_y1)])
                ]

                # Check if the rectangle intersects with the line
                intersects = self.line.intersects(rect_polygon)

                # Check if any corner is touching the line
                touches = any(self.line.touches(Point(coord)) for coord in person_coord)

                # Check if any edge of the bounding box is exactly on the line
                standing = any(
                    line_segment.equals(self.line) or line_segment.intersects(self.line) for line_segment in rect_lines)

                if touches or intersects or standing:
                    count += 1
                    any_intersecting_or_touching_or_standing = True

            return any_intersecting_or_touching_or_standing, count

        except Exception as ex:
            print(ex)
            return False, 0  # Return False and 0 if an exception occurs
