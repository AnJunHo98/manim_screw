import tkinter as tk
import socket, json

HOST, PORT = "127.0.0.1", 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send(payload):
    sock.sendto(json.dumps(payload).encode("utf-8"), (HOST, PORT))

root = tk.Tk()
root.title("ScrewNet Control UI")

# 슬라이더 값 변경 시 전송
def mk_slider(name, lo, hi, init, row, res=0.01):
    var = tk.DoubleVar(value=init)
    def on_change(_):
        send({name: var.get()})
    s = tk.Scale(root, variable=var, from_=lo, to=hi,
                 resolution=res, orient="horizontal", length=420,
                 command=on_change, label=name)
    s.grid(row=row, column=0, padx=10, pady=6, sticky="we")
    return var

yaw   = mk_slider("yaw",   -3.14159,  3.14159, 0.0, 0, res=0.01)
pitch = mk_slider("pitch", -1.45,     1.45,    0.0, 1, res=0.01)
alpha = mk_slider("alpha", -5.0,      5.0,     0.0, 2, res=0.05)
beta  = mk_slider("beta",  -5.0,      5.0,     2.0, 3, res=0.05)

# 모드 라디오 버튼
mode_var = tk.IntVar(value=2)
def on_mode():
    send({"mode": mode_var.get()})

mframe = tk.LabelFrame(root, text="Mode")
mframe.grid(row=4, column=0, padx=10, pady=8, sticky="we")

for k, name in [(1,"Locked"), (2,"Revolute"), (3,"Prismatic"), (4,"Screw")]:
    tk.Radiobutton(mframe, text=f"{k} {name}", value=k,
                   variable=mode_var, command=on_mode).pack(side="left", padx=8, pady=6)

# 초기값 한 번 전송
send({"yaw": yaw.get(), "pitch": pitch.get(), "alpha": alpha.get(), "beta": beta.get(), "mode": mode_var.get()})

root.mainloop()
sock.close()