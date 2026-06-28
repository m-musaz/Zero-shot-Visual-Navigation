from langchain_groq import ChatGroq
from description import Llava_generation

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    api_key="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"
)


def get_image_from_desc(Descriptions,Images,Goal):
    '''
    Descriptions: list of descriptions
    Images: list of images
    Goal: the goal description
    Get the image that best matches the goal 
    '''
    prompt_for_hlp=f"""
        You are a step by step navigator. You are given a list of descriptions you must chose which image contains a path that best matches the description.
        OUPUT ONLY THE NUMBER OF THE IMAGE THAT BEST MATCHES THE DESCRIPTION
"""
    Description_prompts=[]
    for i in range(len(Descriptions)):
        Description_prompts.append(f"Image Description-{i}: {Descriptions[i]}")

    prompt_for_hlp+="\n".join(Description_prompts)
    prompt_for_hlp+=f"\nGoal: {Goal}"

    Chosen_Image_num=llm.invoke(prompt_for_hlp)
    Chosen_Image_num=int(Chosen_Image_num)
    return Images[Chosen_Image_num],Chosen_Image_num

def get_nav_point(Image,Goal):

    prompt="""
     You are a image decriber. You must describe images to help with navigation. The given image contains numbers. 
     These numbers can be used for navigation. You need to describe the scene right next to each number.
     Lets say there are 2 numbers you must format like
     nav-1: scene description about 1
     nav-2:scene description about 2
"""

    desc=Llava_generation(prompt,Image)

    nav_prompt=f"""You are a navigator. You are given a text with navigation numbers.
    These represent the points which you can navigate to in the image. You must pick a navigation number description 
    that best aligns with the goal description.
    Goal: {Goal}
    {desc}
    YOU MUST ANSWER WITH THE NUMBER OF THE NAVIGATION POINT ONLY THAT BEST MATCHES THE GOAL DESCRIPTION.
    YOU MUST RESPOND WITH SOMETHING THAT CAN BE CONVERTED TO AN INTEGER.
    """
    nav_point=llm.invoke(nav_prompt)
    nav_point=int(nav_point)
    return nav_point





