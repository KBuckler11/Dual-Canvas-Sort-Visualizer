import tkinter as tk
from tkinter import ttk, Canvas, IntVar, Frame, StringVar
from ttkbootstrap import Style
import numpy as np
import threading


class DualCanvasSortingVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Dual Canvas Sorting Algorithm Visualizer")
        self.style = Style(theme="darkly")

        self.canvas_controls = {}
        self.control_frame = Frame(master)
        self.control_frame.pack()

        self.setup_shared_controls()
        self.setup_canvas_controls('Canvas 1', 1)
        self.setup_canvas_controls('Canvas 2', 2)

        self.is_paused = False  # Add this line to initialize the pause flag
        self.canvas_states = {
            1: {"array": [], "is_paused": False, "is_sorted": False},
            2: {"array": [], "is_paused": False, "is_sorted": False}
        }
        self.initialize_arrays()  # Note this method name change and implementation

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="Unpause")
        else:
            self.pause_button.config(text="Pause")

    def setup_shared_controls(self):
        self.array_size_var = IntVar(value=50)
        self.array_size_slider = ttk.Scale(self.control_frame, from_=10, to=500, variable=self.array_size_var,
                                           orient="horizontal", command=self.update_array_size_label)

        self.array_size_slider.pack()

        # Fix: Initialize array_size_label as an instance attribute
        # Inside setup_shared_controls
        self.array_size_label = ttk.Label(self.control_frame, text=f"Array Size: {self.array_size_var.get()}")
        self.array_size_label.pack()

        self.speed_var = IntVar(value=50)
        self.speed_slider = ttk.Scale(self.control_frame, from_=0.1, to=100, variable=self.speed_var,
                                      orient="horizontal",
                                      command=self.update_speed_label)
        self.speed_slider.pack()
        self.speed_label = ttk.Label(self.control_frame, text=f"Speed: {self.speed_var.get()} ms")
        self.speed_label.pack()

        self.start_button = ttk.Button(self.control_frame, text="Start Sorting", command=self.start_sorting)
        self.start_button.pack()

        self.shuffle_button = ttk.Button(self.control_frame, text="Shuffle", command=self.shuffle_array)
        self.shuffle_button.pack()

        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack()

    def initialize_arrays(self):
        array_size = self.array_size_var.get()
        array1 = np.linspace(5, 295, array_size, dtype=int)
        np.random.shuffle(array1)
        self.canvas_states[1]["array"] = array1
        self.canvas_states[2]["array"] = array2 = np.array(array1.copy())
        # Update the display for both canvases
        self.display_array(1)
        self.display_array(2)

    # Corrected shuffle_array method
    def shuffle_array(self):
        # Shuffle the arrays for both canvases
        array_size = self.array_size_var.get()
        shuffled_array = np.linspace(5, 295, array_size, dtype=int)
        np.random.shuffle(shuffled_array)

        # Apply the shuffled array to both canvas states and reset sorted flags
        for canvas_id in self.canvas_states:
            self.canvas_states[canvas_id]["array"] = shuffled_array.copy()
            self.canvas_states[canvas_id]["is_paused"] = False  # Optionally reset pause state
            self.canvas_states[canvas_id]["is_sorted"] = False

        # Update the display for both canvases
        self.display_array(1)
        self.display_array(2)

        # Re-enable the "Start Sorting" button
        self.start_button["state"] = "normal"

    def setup_canvas_controls(self, label, canvas_id):
        canvas_frame = Frame(self.master)
        canvas_frame.pack(side='left', expand=True, padx=10)

        canvas_label = ttk.Label(canvas_frame, text=label)
        canvas_label.pack()

        algorithm_var = StringVar()
        algorithm_selection = ttk.Combobox(canvas_frame, textvariable=algorithm_var, state="readonly")
        algorithm_selection['values'] = ['Bubble Sort', 'Insertion Sort', 'Selection Sort', 'Merge Sort', 'Radix Sort',
                                         'Pancake Sort']
        algorithm_selection.pack()

        canvas = Canvas(canvas_frame, width=400, height=300, bg="black")
        canvas.pack()

        self.canvas_controls[canvas_id] = {'canvas': canvas, 'algorithm_var': algorithm_var}

    def display_array(self, canvas_id, highlight=None):
        # Modifications to ensure GUI updates are thread-safe
        if threading.current_thread() is threading.main_thread():
            self._display_array_impl(canvas_id, highlight)
        else:
            self.master.after(0, lambda: self._display_array_impl(canvas_id, highlight))

    def _display_array_impl(self, canvas_id, highlight=None):
        canvas_info = self.canvas_controls[canvas_id]
        canvas = canvas_info['canvas']
        array = self.canvas_states[canvas_id]["array"]  # Use the specific array for this canvas

        canvas.delete("all")  # Clear existing bars

        n = len(array)  # Use the length of the specific array
        total_gap_space = (n - 1) * 2
        available_width = 400 - total_gap_space
        bar_width = available_width / n

        for i, val in enumerate(array):  # Iterate over the specific array
            x0 = i * (bar_width + 2)
            y0 = 300 - (val / max(array) * 290)  # Scale according to the specific array's values
            x1 = x0 + bar_width
            y1 = 300
            fill_color = "dodgerblue"
            if highlight and i in [x[0] for x in highlight]:
                fill_color = [x[1] for x in highlight if x[0] == i][0]
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="")

    # Update the method call for array size update
    def update_array_size_label(self, event=None):
        # Update the label text
        self.array_size_label.config(text=f"Array Size: {self.array_size_var.get()}")
        # Re-initialize arrays for both canvases
        self.initialize_arrays()

    def update_speed_label(self, event=None):
        self.speed_label.config(text=f"Speed: {self.speed_var.get()} ms")

    def start_sorting_simultaneously(self):
        self.bubble_sort_step(1)  # Start on Canvas 1
        self.bubble_sort_step(2)  # Start on Canvas 2

    def bubble_sort_step(self, canvas_id, i=0, j=0, sorted=False):
        array = self.canvas_states[canvas_id]["array"]
        if sorted or i >= len(array) - 1:
            self.display_array(canvas_id)  # Show the final sorted array
            self.canvas_states[canvas_id]["is_sorted"] = True  # Mark as sorted
            self.check_sort_completion()
            return
        if sorted or i >= len(array) - 1:
            self.display_array(canvas_id)  # Show the final sorted array
            return

        if self.is_paused:
            self.master.after(100, lambda: self.bubble_sort_step(canvas_id, i, j, sorted))
            return

        if j < len(array) - i - 1:
            # Highlight the current comparison in green
            self.display_array(canvas_id, highlight=[(j, "green"), (j + 1, "green")])
            if array[j] > array[j + 1]:
                # Swap if needed and highlight in red
                array[j], array[j + 1] = array[j + 1], array[j]
                self.display_array(canvas_id, highlight=[(j, "red"), (j + 1, "red")])
            # Schedule the next comparison within the current bubble sort pass
            self.master.after(self.speed_var.get(), lambda: self.bubble_sort_step(canvas_id, i, j + 1, sorted))
        else:
            # Move to the next pass of bubble sort
            self.master.after(self.speed_var.get(), lambda: self.bubble_sort_step(canvas_id, i + 1, 0, sorted))

    def check_and_swap(self, canvas_id, i, j, N):
        # Fix: Access the correct array using canvas_id
        array = self.canvas_states[canvas_id]["array"]

        # Perform swap if needed and highlight in red
        if array[j] > array[j + 1]:
            array[j], array[j + 1] = array[j + 1], array[j]
            self.display_array(canvas_id, highlight=[(j, "red"), (j + 1, "red")])

        # Regardless of whether a swap occurred, proceed to the next comparison
        j += 1
        self.master.after(self.speed_var.get() // 2, lambda: self.bubble_sort_step(canvas_id, i, j, N == i + 1 + j))

    def insertion_sort_step(self, canvas_id, i=1, j=None, key=None):
        if self.is_paused:
            # If paused, reschedule this step
            self.master.after(100, lambda: self.insertion_sort_step(canvas_id, i, j, key))
            return

        array = self.canvas_states[canvas_id]["array"]

        if i >= len(array):
            # Sorting complete
            self.display_array(canvas_id)
            self.canvas_states[canvas_id]["is_sorted"] = True
            self.check_sort_completion()
            return

        if j is None:
            j = i - 1
            key = array[i]  # Assign 'key' here for the current 'i'

        # Now, 'key' is always initialized for the operations below.
        if j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            self.display_array(canvas_id, highlight=[(j + 1, "red"), (i, "green")])
            j -= 1
            # Pass 'key' along to ensure it's available for subsequent steps.
            self.master.after(self.speed_var.get(), lambda: self.insertion_sort_step(canvas_id, i, j, key))
        else:
            array[j + 1] = key
            self.display_array(canvas_id, highlight=[(j + 1, "green")])
            # Move to the next 'i', reset 'j' to None, and don't need to pass 'key' as it will be reassigned.
            self.master.after(self.speed_var.get(), lambda: self.insertion_sort_step(canvas_id, i + 1))

    def selection_sort_step(self, canvas_id, start=0, i=None, min_idx=None):
        array = self.canvas_states[canvas_id]["array"]
        n = len(array)

        if start >= n - 1:
            # Sorting complete
            self.canvas_states[canvas_id]["is_sorted"] = True
            self.check_sort_completion()
            return

        if i is None:
            i = start
            min_idx = start

        if i < n:
            # Highlight the current and min elements
            highlights = [(min_idx, "red")]
            if i != min_idx:
                highlights.append((i, "green"))
            self.display_array(canvas_id, highlight=highlights)

            if array[i] < array[min_idx]:
                min_idx = i

            if self.is_paused:
                self.master.after(100, lambda: self.selection_sort_step(canvas_id, start, i + 1, min_idx))
            else:
                self.master.after(self.speed_var.get(),
                                  lambda: self.selection_sort_step(canvas_id, start, i + 1, min_idx))
        else:
            # Swap the found minimum element with the first element
            array[start], array[min_idx] = array[min_idx], array[start]
            self.display_array(canvas_id, highlight=[(start, "green"), (min_idx, "red")])
            self.master.after(self.speed_var.get(), lambda: self.selection_sort_step(canvas_id, start + 1))

    def merge_sort_init(self, canvas_id):
        end = len(self.canvas_states[canvas_id]["array"])
        self.merge_sort(canvas_id, 0, end)
        self.canvas_states[canvas_id]["is_sorted"] = True  # Mark as sorted once done
        self.check_sort_completion()

    def merge_sort(self, canvas_id, start, end):
        if end - start > 1:
            mid = (start + end) // 2
            self.merge_sort(canvas_id, start, mid)  # Sort the first half
            self.merge_sort(canvas_id, mid, end)  # Sort the second half
            self.merge(canvas_id, start, mid, end)  # Merge the sorted halves

    def merge(self, canvas_id, start, mid, end):
        temp_array = []
        left, right = start, mid

        # Build the temp array for the merge process
        while left < mid and right < end:
            if self.canvas_states[canvas_id]["array"][left] <= self.canvas_states[canvas_id]["array"][right]:
                temp_array.append(self.canvas_states[canvas_id]["array"][left])
                left += 1
            else:
                temp_array.append(self.canvas_states[canvas_id]["array"][right])
                right += 1

        temp_array.extend(self.canvas_states[canvas_id]["array"][left:mid])
        temp_array.extend(self.canvas_states[canvas_id]["array"][right:end])

        # Apply the merged array back to the original array and visualize this step
        for i, val in enumerate(temp_array):
            self.canvas_states[canvas_id]["array"][start + i] = val
            # Visualize the update immediately, no scheduling to avoid breaking the sort
            self.display_array(canvas_id, highlight=[(start + i, "red")])
            self.master.update_idletasks()  # Force GUI update for each step
            self.master.after(self.speed_var.get())

    def counting_sort_for_radix(self, canvas_id, exp, start_update=0):
        array = self.canvas_states[canvas_id]["array"]
        n = len(array)
        output = [0] * n
        count = [0] * 10

        # Store count of occurrences in count[]
        for i in range(n):
            index = (array[i] // exp) % 10
            count[index] += 1

        # Change count[i] so that count[i] now contains actual
        # position of this digit in output array
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build the output array
        i = n - 1
        while i >= 0:
            index = (array[i] // exp) % 10
            output[count[index] - 1] = array[i]
            count[index] -= 1
            i -= 1

        # Apply the sorted output back to the original array and visualize each step
        def update_step(i):
            if i < n:
                array[i] = output[i]
                self.display_array(canvas_id, highlight=[(i, "red")])
                # Schedule the next update
                self.master.after(self.speed_var.get(), lambda: update_step(i + 1))
            else:
                # Once all updates for this digit are visualized, proceed to the next digit
                next_exp = exp * 10
                if max(array) // next_exp > 0:
                    self.counting_sort_for_radix(canvas_id, next_exp)

        update_step(start_update)

    def radix_sort(self, canvas_id):
        self.counting_sort_for_radix(canvas_id, exp=1)

    def flip(self, canvas_id, k, callback):
        array = self.canvas_states[canvas_id]["array"]

        def flip_step(left, right):
            if left < right:
                array[left], array[right] = array[right], array[left]
                self.display_array(canvas_id, highlight=[(left, "red"), (right, "red")])
                # Continue flipping after a delay
                self.master.after(self.speed_var.get(), lambda: flip_step(left + 1, right - 1))
            else:
                # Once the flip is complete, proceed to the next step
                callback()

        flip_step(0, k)

    def pancake_sort(self, canvas_id):
        array = self.canvas_states[canvas_id]["array"]
        n = len(array)

        def sort_step(n):
            if self.is_paused:
                # If paused, wait a bit and then continue sorting
                self.master.after(100, lambda: sort_step(n))
                return

            if n <= 1:
                # Final display to show the sorted array
                self.display_array(canvas_id)
                return

            mi = self.max_index(canvas_id, n)
            if mi != n - 1:
                # Flip the maximum number to the beginning
                self.flip(canvas_id, mi, lambda:
                # Then flip it to its correct position
                self.flip(canvas_id, n - 1, lambda: sort_step(n - 1))
                          )
            else:
                sort_step(n - 1)

        sort_step(n)

    def max_index(self, canvas_id, n):
        array = self.canvas_states[canvas_id]["array"]
        mi = 0
        for i in range(1, n):
            # Visualization update for comparison
            self.display_array(canvas_id, highlight=[(mi, "green"), (i, "green")])
            # Ensure visualization delay respects user's speed setting
            self.master.after(self.speed_var.get(), lambda: None)
            self.master.update_idletasks()

            if array[i] > array[mi]:
                mi = i
        return mi

        def sort_step(i):
            if i <= 1:
                self.display_array(canvas_id)  # Final display to show the sorted array
                return
            mi = self.max_index(canvas_id, i)
            if mi != i - 1:
                self.flip(canvas_id, mi, lambda: self.flip(canvas_id, i - 1, lambda: sort_step(i - 1)))
            else:
                sort_step(i - 1)

        sort_step(n)

    def start_sorting(self):
        self.start_button["state"] = "disabled"
        for canvas_id in self.canvas_states:
            algorithm = self.canvas_controls[canvas_id]['algorithm_var'].get()
            if algorithm == "Merge Sort":
                thread = threading.Thread(target=lambda: self.merge_sort_init(canvas_id))
                thread.start()

            elif algorithm == "Bubble Sort":
                self.bubble_sort_step(canvas_id)
            elif algorithm == "Insertion Sort":
                self.insertion_sort_step(canvas_id)
            elif algorithm == "Selection Sort":
                self.selection_sort_step(canvas_id)
            elif algorithm == "Radix Sort":
                thread = threading.Thread(target=lambda: self.radix_sort(canvas_id), daemon=True)
                thread.start()
            elif algorithm == "Pancake Sort":
                self.pancake_sort(canvas_id)

    def check_sort_completion(self):
        if all(self.canvas_states[canvas_id]["is_sorted"] for canvas_id in self.canvas_states):
            # Re-enable the start button if sorting is complete on both canvases
            self.start_button["state"] = "normal"

    def is_array_sorted(self, array):
        """Check if the given array is sorted."""
        return all(array[i] <= array[i + 1] for i in range(len(array) - 1))

    def check_sort_completion(self):
        if all(self.canvas_states[canvas]["is_sorted"] for canvas in self.canvas_states):
            self.start_button["state"] = "normal"


if __name__ == '__main__':
    root = tk.Tk()
    app = DualCanvasSortingVisualizer(root)
    root.mainloop()
