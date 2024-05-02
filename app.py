import tkinter as tk
from PIL import Image, ImageTk
from adaptive_teaching import *
from utils import *


class FlashcardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Learning App")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        self.start_screen()

    def start_screen(self):
        self.clear_widgets()

        start_label = tk.Label(self.root, text="Welcome to Adaptive Learning App", font=("Helvetica", 20))
        start_label.pack(pady=20) 

        start_button = tk.Button(self.root, text="Start", command=self.start_flashcards)
        start_button.pack()

    def start_flashcards(self):
        self.clear_widgets()
        self.m = 15  # Time 'T'
        self.th = [2, 1, 0]
        self.epsilon = 0.1
        self.reward_type = "continuous"  # binary or continuous
        self.algo = "epsilon-greedy"  # epsilon-greedy or gd
        self.n_y, self.algo = get_parameters(self.reward_type, self.algo)
        self.concepts = ['Gorilla', 'Panda', 'Wolf', 'Lion', 'Elephant', 'Tiger', 'Giraffe', 'Dolphin', 'Penguin', 'Koala',
                    'Polar bear', 'Kangaroo', 'Cheetah', 'Zebra', 'Rhino', 'Quokka', 'Pangolin', 'Sun bear', 'Numbat',
                    'Quoll', 'Maned wolf', 'Pika', 'Shoebill', 'Serval', 'Markhor', 'Tufted Deer', 'Binturong',
                    'Dik dik', 'Tarsier', 'Red Uakari']
        self.n = len(self.concepts)
        self.start=10
        self.theta = generate_shared_theta(self.th, self.n)
        self.sigma_t_i, self.y_t_i = generate_random_lists(self.concepts, self.start, self.reward_type)

        self.flashcards =  [
            {"image_path": "c1.jpg", "answer": "Gorilla"},
            {"image_path": "c2.jpg", "answer": "Panda"},
            {"image_path": "c3.jpeg", "answer": "Wolf"},
            {"image_path": "c4.jpeg", "answer": "Lion"},
            {"image_path": "c5.jpg", "answer": "Elephant"},
            {"image_path": "c6.jpg", "answer": "Tiger"},
            {"image_path": "c7.jpg", "answer": "Giraffe"},
            {"image_path": "c8.jpg", "answer": "Dolphin"},
            {"image_path": "c9.jpeg", "answer": "Penguin"},
            {"image_path": "c10.jpg", "answer": "Koala"},
            {"image_path": "c11.jpg", "answer": "Polar bear"},
            {"image_path": "c12.jpg", "answer": "Kangaroo"},
            {"image_path": "c13.jpg", "answer": "Cheetah"},
            {"image_path": "c14.jpg", "answer": "Zebra"},
            {"image_path": "c15.jpg", "answer": "Rhino"},
            {"image_path": "c16.jpg", "answer": "Quokka"},
            {"image_path": "c17.jpg", "answer": "Pangolin"},
            {"image_path": "c18.jpeg", "answer": "Sun bear"},
            {"image_path": "c19.jpg", "answer": "Numbat"},
            {"image_path": "c20.jpeg", "answer": "Quoll"},
            {"image_path": "c21.jpg", "answer": "Maned wolf"},
            {"image_path": "c22.jpg", "answer": "Pika"},
            {"image_path": "c23.jpg", "answer": "Shoebill"},
            {"image_path": "c24.jpeg", "answer": "Serval"},
            {"image_path": "c25.jpg", "answer": "Markhor"},
            {"image_path": "c26.jpg", "answer": "Tufted Deer"},
            {"image_path": "c27.jpg", "answer": "Binturong"},
            {"image_path": "c28.jpg", "answer": "Dik dik"},
            {"image_path": "c29.jpg", "answer": "Tarsier"},
            {"image_path": "c30.jpg", "answer": "Red Uakari"}
        ]

        self.current_flashcard_index = 0

        self.flashcard_image_label = tk.Label(self.root)
        self.flashcard_image_label.pack()

        self.answer_entry = tk.Entry(self.root)
        self.answer_entry.pack()

        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit_answer)
        self.submit_button.pack()

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

        self.timer_label = tk.Label(self.root, text="")
        self.timer_label.pack()

        self.photo = None

        self.timer_id = None

        self.show_next_flashcard()

    def show_next_flashcard(self):
        self.result_label.config(text="")
        self.answer_entry.delete(0, tk.END)

        if self.current_flashcard_index < (self.m-self.start):
            if self.algo == "epsilon-greedy":
                self.i_t = epsgreeedy_i(self.sigma_t_i, self.y_t_i, self.n, self.concepts, len(self.flashcards), self.theta, self.n_y,
                               self.current_flashcard_index, self.epsilon)
            elif self.algo == "gd":
                self.i_t = argmax_i(self.sigma_t_i, self.y_t_i, self.n, self.concepts, len(self.flashcards), self.theta, self.n_y,
                           self.current_flashcard_index)
            print("Concept Recommended: ", self.i_t)
            self.sigma_t_i = self.sigma_t_i + [self.i_t]
            print("Updated Sigma: ", self.sigma_t_i)

            self.flashcard_idx = get_flashcard_idx(self.flashcards, self.i_t)
            flashcard = self.flashcards[self.flashcard_idx]
            image_path = flashcard["image_path"]

            image = Image.open("Animals/" + image_path)
            image = image.resize((400, 300))
            self.photo = ImageTk.PhotoImage(image)
            self.flashcard_image_label.config(image=self.photo)

            self.start_timer(10)

            self.current_flashcard_index += 1
        else:
            self.end_screen()

    def start_timer(self, seconds):
        self.update_timer(seconds)

        if seconds > 0:
            self.timer_id = self.root.after(1000, self.start_timer, seconds - 1)
        else:
            flashcard = self.flashcards[self.flashcard_idx]
            correct_answer = flashcard["answer"]
            self.result_label.config(text="Time's up! Correct answer: {}".format(correct_answer), fg="red")
            y_t_new = 0
            self.y_t_i.append(y_t_new)
            print("Updated y: ", self.y_t_i)
            self.root.after(3000, self.show_next_flashcard_with_wait)

    def update_timer(self, seconds):
        self.timer_label.config(text="Time left: {}s".format(seconds))

    def submit_answer(self):
        user_answer = self.answer_entry.get()
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            flashcard = self.flashcards[self.flashcard_idx]
            correct_answer = flashcard["answer"]
            if self.reward_type == "continuous":
                y_t_new = fuzzy_match(self.i_t, [user_answer])
            else:
                y_t_new = check_answer(self.i_t, user_answer)
            self.y_t_i.append(y_t_new)
            print("Updated y: ", self.y_t_i)
            self.result_label.config(text="Correct answer: {}".format(correct_answer), fg="blue")
            self.root.after(3000, self.show_next_flashcard_with_wait)

    def show_next_flashcard_with_wait(self):
        self.result_label.config(text="Next lesson will start in 3 seconds.")
        self.timer_label.config(text="")
        self.root.after(3000, self.show_next_flashcard)

    def end_screen(self):
        self.clear_widgets()
        self.fielname=get_filename()
        np.save("Results/"+self.fielname+".npy", np.array(self.y_t_i))
        end_label = tk.Label(self.root, text="Congratulations! \n You have completed all flashcards." + "\n"+ "Pre-Learning Score: "+str(sum(self.y_t_i[:self.start])/len(self.y_t_i[:self.start]))+ "\n"+ "Post-Learning Score: "+str(sum(self.y_t_i[self.start:])/len(self.y_t_i[self.start:])), font=("Helvetica", 20))
        end_label.pack(pady=20)

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()


root = tk.Tk()

app = FlashcardApp(root)

root.mainloop()
