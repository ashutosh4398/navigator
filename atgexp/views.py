from django.shortcuts import render
from .models import Avatar
from .serializers import ImageSerializers
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser,FormParser, MultiPartParser
from django.http import JsonResponse
from rest_framework import status
from . import yolo
from .apps import AtgexpConfig
import os
from django.conf import settings
import random as r

from itertools import chain
from nltk.corpus import wordnet

# Create your views here.

class ImageClass(APIView):

	parser_classes = (JSONParser, FormParser, MultiPartParser)
	

	def post(self,request):
		serializer = ImageSerializers(data = request.data)
		if(serializer.is_valid()):
			serializer.save()
			obj = Avatar.objects.all()[0]
			# object detection code starts
			
			image_path = os.path.join(settings.MEDIA_ROOT,str(obj.image))
			
			try:
	 			yolo_result = yolo.predict(obj.image)
	 			gru_result,attention_plot = AtgexpConfig.evaluate(image_path)
	 			
	 			if ('<unk>' in gru_result):
	 				gru_result.remove('<unk>')
	 			
	 			gru_result.remove(gru_result[-1])
	 					
					
	 			flag = False	 				 			
	 			for each in set(yolo_result):
	 				if(each in gru_result):
	 					flag = True
	 					break
	 				else:
	 					synonyms = wordnet.synsets(each)
	 					lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
	 					print(lemmas)
	 					if(each in lemmas):
	 						flag = True
	 						break
	 			if(flag):
	 				ans = ' '.join(gru_result)
	 			else:
	 				
	 				if(len(yolo_result) > 1):
		 				ans = ' '.join(yolo_result)
		 				
		 				sentences = [
							f"{ans} objects are in front of you",
							f"I can see {ans} ahead",
							f"Maybe there are {ans} exactly infront of you",						
						]
		 				ans = r.sample(sentences,k=1)
		 			else:
		 				ans = "Unable to detect"
	 					
	 			
			
			except Exception as e:
				print(e)
			else:
				print(yolo_result)
				print(ans)
				os.remove(image_path)
			finally:
				obj.delete()
			
			# object detection code ends
			# delete the image object from database
			

			return JsonResponse({
				'caption': ans
			},safe=False)
		else:
			return Response(serializer.errors, status=400)
		

