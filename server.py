from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
class Handler(SimpleHTTPRequestHandler):
 def end_headers(self):
  self.send_header('Cache-Control','no-store')
  super().end_headers()
ThreadingHTTPServer(('127.0.0.1',7870),Handler).serve_forever()
