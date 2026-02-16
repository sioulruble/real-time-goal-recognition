import socket
import numpy as np

def parse_eye_msg(s:str):
	s=s.strip()
	if "#" not in s:
		return None,None
	left, right = s.split("#",1)
	px,py,pz = map(float,left.strip().split())
	dx,dy,dz = map(float,right.strip().split())
	gaze_pos = np.array([px,py,pz],float)
	#gaze_dir = np.array([dx,dy,dz],float)/np.linalg.norm([dx,dy,dz])
	gaze_dir = np.array([dx,dy,dz],float)
	return gaze_pos


class EyesTracking:
	def __init__(self, port=9999, max_port_tries=10):
		"""
		Bind the server socket. If requested port is busy, try an ephemeral port.
		"""
		self.port = port
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		bound = False
		try:
			self.server_socket.bind(("0.0.0.0", self.port))
			bound = True
		except OSError as e:
			print(f"[EyesTracking] Port {self.port} busy: {e}. Trying to bind an ephemeral port...")
			try:
				self.server_socket.bind(("0.0.0.0", 0))  # ask OS for free port
				self.port = self.server_socket.getsockname()[1]
				bound = True
				print(f"[EyesTracking] Bound to ephemeral port {self.port}")
			except Exception as e2:
				print(f"[EyesTracking] Failed to bind ephemeral port: {e2}")

		if not bound:
			self.server_socket.close()
			raise RuntimeError(f"Could not bind EyesTracking socket to port {port} or an ephemeral port.")

		self.server_socket.listen(1)
		self.conn = None
		self.addr = None
		self.latest_data = None
		self.running = True
		self.gaze_position = None
		
	def stream_data(self):
		print(f"[SERVER] Listening on 0.0.0.0:{self.port}")
		while self.running:
			print("[SERVER] Waiting for connection...")
			self.conn, self.addr = self.server_socket.accept()
			print("[SERVER] Connected to:", self.addr)
			try:
				while self.running:
					data = self.conn.recv(1024)
					if not data:
						break
					decoded = data.decode().strip()
					self.latest_data = decoded
					self.gaze_position = parse_eye_msg(decoded)
					print("gaze pos:", self.gaze_position)
			except Exception as e:
				print("[SERVER] Erreur:", e)
			finally:
				self.conn.close()
				self.conn = None		
	def stop(self):
		self.running = False
		self.conn.close()
		self.server_socket.close()


def find_gaze_value(camera):
    gaze_position=camera.eye_tracker.gaze_position
    return gaze_position

def convert_gaze(frame,gaze_position):
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    h,w = frame.shape[:2]
    u,v= float(gaze_position[0]), float(gaze_position[1])
    x = int(u*w)
    y = int((1-v)*h)
    return x,y


