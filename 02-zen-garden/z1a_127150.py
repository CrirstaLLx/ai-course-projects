# import tkinter as tk
# import random
#
# CELL_SIZE = 40  # veľkosť štvorca v pixeloch
#
# class ZenGardenApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Zen Garden Generator")
#
#         # --- Rámec pre vstupné hodnoty ---
#         input_frame = tk.Frame(master)
#         input_frame.pack(pady=10)
#
#         tk.Label(input_frame, text="Width:").grid(row=0, column=0)
#         tk.Label(input_frame, text="Height:").grid(row=0, column=2)
#         tk.Label(input_frame, text="Stones:").grid(row=0, column=4)
#
#         self.width_entry = tk.Entry(input_frame, width=5)
#         self.height_entry = tk.Entry(input_frame, width=5)
#         self.stone_entry = tk.Entry(input_frame, width=5)
#
#         self.width_entry.grid(row=0, column=1)
#         self.height_entry.grid(row=0, column=3)
#         self.stone_entry.grid(row=0, column=5)
#
#         # --- Tlačidlo na generovanie ---
#         self.generate_button = tk.Button(master, text="Generate Garden", command=self.generate_garden)
#         self.generate_button.pack(pady=10)
#
#         # --- Plátno pre vykreslenie záhradky ---
#         self.canvas = tk.Canvas(master, bg="beige")
#         self.canvas.pack()
#
#         self.grid = []  # sem uložíme aktuálnu záhradku
#
#     def generate_garden(self):
#         try:
#             width = int(self.width_entry.get())
#             height = int(self.height_entry.get())
#             stone_count = int(self.stone_entry.get())
#
#             if width <= 0 or height <= 0:
#                 raise ValueError("Rozmery musia byť kladné čísla.")
#             if stone_count >= width * height:
#                 raise ValueError("Príliš veľa kameňov pre danú veľkosť záhradky.")
#
#         except ValueError as e:
#             self.show_error(str(e))
#             return
#
#         # vytvor novú záhradku
#         self.grid = [[0 for _ in range(width)] for _ in range(height)]
#
#         # náhodné umiestnenie kameňov
#         stones = random.sample([(y, x) for y in range(height) for x in range(width)], stone_count)
#         for (y, x) in stones:
#             self.grid[y][x] = "K"
#
#         # vykresli
#         self.draw_grid(width, height)
#
#     def draw_grid(self, width, height):
#         self.canvas.delete("all")
#         self.canvas.config(width=width * CELL_SIZE, height=height * CELL_SIZE)
#
#         for y in range(height):
#             for x in range(width):
#                 value = self.grid[y][x]
#                 color = "gray" if value == "K" else "white"
#                 self.canvas.create_rectangle(
#                     x * CELL_SIZE, y * CELL_SIZE,
#                     (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
#                     fill=color, outline="black"
#                 )
#                 if value != 0:
#                     self.canvas.create_text(
#                         x * CELL_SIZE + CELL_SIZE / 2,
#                         y * CELL_SIZE + CELL_SIZE / 2,
#                         text=value,
#                         font=("Arial", 14, "bold"),
#                         fill="black"
#                     )
#
#     def show_error(self, message):
#         error_window = tk.Toplevel(self.master)
#         error_window.title("Error")
#         tk.Label(error_window, text=message, fg="red", padx=20, pady=10).pack()
#         tk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=5)
#
#
# # --- Spustenie aplikácie ---
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ZenGardenApp(root)
#     root.mainloop()
