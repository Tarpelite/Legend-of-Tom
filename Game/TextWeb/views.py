from django.shortcuts import render
from django.views.decorators import csrf
from django.http import HttpResponse
from django.template import loader
from django.contrib import messages
from django.utils.safestring import mark_safe
import re

from . import Worker

worker = Worker.GPTWorker()
ans_num = 4
# Create your views here.

def Answer(request):
    '''
        Accoring to the text, give 4 answers
    '''
    context = {}
    Keywords = ["He", "he", "his","His", "She", "she","Her","her","I have", "I am", "My", "my", "in", "to"]
    if request.POST:
        print(request.POST.keys())
        print(len(request.POST['time']))
        prompt = request.POST['prompt'].strip()
        print(len(prompt))
        ans = worker.predict_one(prompt)
        ans_tokens = list(ans.split(' '))
        flag = False
        stop = None
        for keyword in Keywords:
            if keyword in ans_tokens:
                choice1 = ans
                flag = True
                stop = keyword
                break
        if flag:
            original_ans = ans
            ans = original_ans.split(stop)[0]
            choice1 = stop + ' ' + ' '.join(original_ans.split(stop)[1:])
            choices = worker.predict_choices(prompt + ' ' + ans, ans_num - 1)
            choices.append(choice1)
            context['Choices'] = choices
        context['ans'] = prompt + ' ' + ans
        print('ans', ans)
    return render(request, 'Bot.html', context=context)

def Choice(request):
    context = {}
    if request.POST:
        print(request.POST)
        prompt = request.POST['prompt'].strip()
        choice = request.POST['choice'].strip()
        context['ans'] = prompt + choice
    return render(request, 'Bot.html', context=context)

def predict_based_time(request):
    '''
        According to the time and text, make the predictions
    '''
    context = {}
    context['width'] = 0
    context['time'] = 1950
    if request.POST:
        if len(request.POST['time']) == 0:
            context['time'] = 1950
            time = 1950
        else:
            time = int(request.POST['time'].strip())
            time += 10
            context['time'] = time
        prompt = request.POST['prompt'].strip()
        choices = worker.time_based_predict(prompt, time)
        width = (time - 1931)*1.00/(2000 - 1931)
        width = width*100
        context['width'] = width
        if time == 2000:
            context['end'] = "Yes"
            context['ans'] = prompt + "\n" + "In 2000, He came to the end of his life."
            return render(request, 'Bot.html', context=context)
        width = (time - 1931)*1.00/(2000 - 1931)
        width = width*100
        context['width'] = width
        context['ans'] = prompt
        context['Choices'] = choices
    return render(request, 'Bot.html', context=context)
             

        












