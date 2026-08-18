from route_client import get_route, classify_segments, route_city_fraction

chennai = (80.2707, 13.0827)
bangalore = (77.5946, 12.9716)

route = get_route(chennai, bangalore)
segments = classify_segments(route)

print("Number of segments:", len(segments))
print("City fraction:", route_city_fraction(segments))