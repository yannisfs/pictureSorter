# folder_chooser_gui.py

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk, messagebox
import threading
from picture_sorter import PictureSorter

def choose_folder(entry):
    """Open a folder selection dialog and insert the selected path into the entry."""
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry.delete(0, tk.END)  # Clear current content
        entry.insert(0, folder_selected)  # Insert the selected folder path

def update_progress(value):
    """Update the progress bar."""
    progress['value'] = value
    root.update_idletasks()

def sort_images_thread(source, destination, sorter):
    """Thread target function to sort images."""
    try:
        sorter.sort_images(source, destination, n_clusters=get_cluster_count())
        messagebox.showinfo("Success", f"Images have been sorted successfully!\nSaved in: {destination}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")
    finally:
        stop_progress()
        progress['value'] = 0  # Reset progress bar
        status_label.config(text="Completed.", foreground="green")
        confirm_button.config(state='normal')  # Re-enable the Confirm button

def get_cluster_count():
    """Retrieve the number of clusters based on user input."""
    if specify_clusters_var.get():
        try:
            count = int(cluster_spinbox.get())
            if count <= 0:
                raise ValueError
            return count
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for clusters.")
            raise
    else:
        return 5  # Default number of clusters

def confirm_action():
    """Action to perform when the Confirm button is clicked."""
    folder1 = folder_entry1.get()
    folder2 = folder_entry2.get()

    if not folder1 or not folder2:
        status_label.config(text="Please select both folders.", foreground="red")
        return

    if folder1 == folder2:
        status_label.config(text="Source and Destination folders must be different.", foreground="red")
        return

    # Disable the Confirm button to prevent multiple clicks
    confirm_button.config(state='disabled')
    status_label.config(text="Processing...", foreground="blue")

    # Initialize the PictureSorter
    sorter = PictureSorter()

    # Start the sorting process in a separate thread
    thread = threading.Thread(
        target=sort_images_thread,
        args=(folder1, folder2, sorter),
        daemon=True
    )
    thread.start()

def on_closing():
    """Handle the window closing event."""
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

def toggle_cluster_spinbox():
    """Enable or disable the cluster spinbox based on the checkbox."""
    if specify_clusters_var.get():
        cluster_spinbox.config(state='normal')
    else:
        cluster_spinbox.config(state='disabled')

def start_progress():
    """Start the progress bar animation."""
    progress.start(10)  # Move 10ms steps

def stop_progress():
    """Stop the progress bar animation."""
    progress.stop()

# Create the main application window
root = tk.Tk()
root.title("Picture Sorter")
root.geometry("650x300")  # Increased size for better layout

# Configure grid layout with padding
root.columnconfigure(0, weight=1, pad=10)
root.columnconfigure(1, weight=3, pad=10)
root.columnconfigure(2, weight=1, pad=10)
root.rowconfigure([0, 1, 2, 3, 4, 5, 6], pad=10)

# Style configuration
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))
style.configure('TEntry', font=('Helvetica', 10))
style.configure('TSpinbox', font=('Helvetica', 10))
style.configure('Horizontal.TProgressbar', troughcolor='white')

# Folder Chooser 1 - Source Folder
folder_label1 = ttk.Label(root, text="Source Folder:")
folder_label1.grid(row=0, column=0, sticky=tk.E)

folder_entry1 = ttk.Entry(root, width=50)
folder_entry1.grid(row=0, column=1, sticky=tk.W)

browse_button1 = ttk.Button(root, text="Browse...", command=lambda: choose_folder(folder_entry1))
browse_button1.grid(row=0, column=2, sticky=tk.W)

# Folder Chooser 2 - Destination Folder
folder_label2 = ttk.Label(root, text="Destination Folder:")
folder_label2.grid(row=1, column=0, sticky=tk.E)

folder_entry2 = ttk.Entry(root, width=50)
folder_entry2.grid(row=1, column=1, sticky=tk.W)

browse_button2 = ttk.Button(root, text="Browse...", command=lambda: choose_folder(folder_entry2))
browse_button2.grid(row=1, column=2, sticky=tk.W)

# Optional: Specify Number of Clusters
specify_clusters_var = tk.BooleanVar()
specify_clusters_check = ttk.Checkbutton(
    root,
    text="Specify Number of Clusters:",
    variable=specify_clusters_var,
    command=toggle_cluster_spinbox
)
specify_clusters_check.grid(row=2, column=1, sticky=tk.W)

cluster_spinbox = ttk.Spinbox(
    root,
    from_=1,
    to=100,
    width=5,
    state='disabled'  # Disabled by default
)
cluster_spinbox.set(5)  # Default value
cluster_spinbox.grid(row=2, column=1, sticky=tk.E)

# Confirm Button
confirm_button = ttk.Button(root, text="Confirm", command=confirm_action)
confirm_button.grid(row=3, column=1, pady=10)

# Status Label
status_label = ttk.Label(root, text="")
status_label.grid(row=4, column=0, columnspan=3)

# Progress Bar
progress = ttk.Progressbar(
    root,
    orient='horizontal',
    length=400,
    mode='indeterminate',
    style='Horizontal.TProgressbar'
)
progress.grid(row=5, column=0, columnspan=3, pady=10)

# Handle window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the application
root.mainloop()