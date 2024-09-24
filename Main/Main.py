import os
import kivy
from kivy.lang import Builder
from kivy.app import App
from kivy.uix import *
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import time


class Layout(BoxLayout):
    pass
        

class AndriodApp(App):
    def build(self):
        #self.screen_manager = ScreenManager()

        #firstpage = Layout()
        #screen1 = Screen(name='welcomepage')
        #screen1.add_widget(firstpage)
        #self.screen_manager.add_widget(screen1)

        return Builder.load_file('mymain.kv')
    
if __name__ == '__main__':
    My_app = AndriodApp()
    My_app.run()

