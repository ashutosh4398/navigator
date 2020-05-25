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
from rest_framework.response import Response

# Create your views here.

class ImageClass(APIView):

	parser_classes = (JSONParser, FormParser, MultiPartParser)
	

	def post(self,request):
		ans = 'Unable to detect'
		
		serializer = ImageSerializers(data = request.data)
		if(serializer.is_valid()):
			serializer.save()
			obj = Avatar.objects.all().last()
			# object detection code starts
			
			image_path = os.path.join(settings.MEDIA_ROOT,str(obj.image))
			
			try:
	 			yolo_result = yolo.predict(obj.image)
	 			gru_result,attention_plot = AtgexpConfig.evaluate(image_path)
	 			print(f"yolo {set(yolo_result)}")
	 			print(f"gru {gru_result}")
	 			while('<unk>' in gru_result):
	 				gru_result.remove('<unk>')
	 			
	 			gru_result.remove(gru_result[-1])

					
	 			flag = False
	 			count = 0
	 			for each in set(yolo_result):
	 				if(each in gru_result):
# 	 					flag = True
	 					count += 1 
# 	 					break
	 				else:
	 					# synonyms
	 					synonyms = wordnet.synsets(each)
	 					lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
	 					print(lemmas)
	 					if(each in lemmas):
# 	 						flag = True
	 						count += 1
	 				print(count)		
	 				if(count >= len(yolo_result)//2 and len(yolo_result) != 1):
	 					flag = True
	 					break
	 			

	 			if(flag):
	 				ans = ' '.join(gru_result)
	 			else:
	 				
	 				if(len(yolo_result) >= 1):
		 				
		 				temp = []
		 				for x in set(yolo_result):
		 					if(yolo_result.count(x) > 1):
		 						temp.append(f"many {x}")
		 					else:
		 						temp.append(f"a {x}")
		 						
		 				if(len(temp) >= 2):
		 					temp.insert(-1, 'and')
		 				yolo_result = temp
		 				
		 				
		 				ans = ' '.join(yolo_result)
		 				sentences = [
							f"{ans + ' are'  if(len(yolo_result) > 1) else ans + ' is'} in front of you",
							f"I can see {ans} ahead",
							f"Maybe there {'are '+ ans if(len(yolo_result) > 1) else 'is '+ans} exactly infront of you",					
						]
		 				ans = r.sample(sentences,k=1)[0]
		 				print(f"Here => {ans}")
# 		 			else:
# 		 				ans = "Unable to detect"
		 		print(ans)
			except Exception as e:
				print(e)
			else:
				print(yolo_result)
				print(ans)
# 				os.remove(image_path)
			finally:
				obj.caption = ans
				obj.save()
				
			
			# object detection code ends
			# delete the image object from database
			
			return JsonResponse({
				'caption': ans
			},safe=False)
		else:
			return Response(serializer.errors, status=400)
		

