#!/usr/bin/env python3
"""Late-Life argument-level joint planning in one seven-register chronology."""
from __future__ import annotations
from reference_life_extension_recursive_self_culture_control_v1 import A1,A2,B1,B2,Q
from reference_organism_v2_core import INQUIRY_CONTEXT
LIFE_AFTER=('reference_life_extension_recursive_self_culture_control_v1',)
R1,R2,R3=0xF851,0xF852,0xF853
ROLE_FORMS={
'en':{
'b1_other':('you take the sheltered route now','please take the sheltered route','you take the sheltered route together'),
'b1_self':("I'll take the sheltered route now","I'll take the sheltered route first","I'll take the sheltered route with you"),
'a1_self':("I'll inspect the seal now","I'll inspect the seal first","I'll inspect the seal with you"),
'a2_self':("I'll test the latch instead","I'll test the latch next","I'll test the latch with you"),
'b2_other':('then you secure the cover now','then please secure the cover','then you secure the cover together'),
'b2_self':("I'll secure the cover now","I'll secure the cover next","I'll secure the cover with you")},
'de':{
'b1_other':('du nimmst jetzt den geschützten Weg','du nimmst zuerst den geschützten Weg','du nimmst gemeinsam mit mir den geschützten Weg'),
'b1_self':('ich nehme jetzt den geschützten Weg','ich nehme zuerst den geschützten Weg','ich nehme gemeinsam den geschützten Weg'),
'a1_self':('ich prüfe jetzt die Dichtung','ich prüfe zuerst die Dichtung','ich prüfe gemeinsam die Dichtung'),
'a2_self':('ich teste stattdessen die Verriegelung','ich teste als Nächstes die Verriegelung','ich teste gemeinsam die Verriegelung'),
'b2_other':('danach sicherst du die Abdeckung','anschließend sicherst du die Abdeckung','danach sicherst du gemeinsam die Abdeckung'),
'b2_self':('danach sichere ich die Abdeckung','anschließend sichere ich die Abdeckung','ich sichere danach die Abdeckung')},
'ru':{
'b1_other':('ты сейчас выбираешь защищённый путь','сначала ты выбираешь защищённый путь','ты теперь выбираешь защищённый путь'),
'b1_self':('я сейчас выбираю защищённый путь','сначала я выбираю защищённый путь','я теперь выбираю защищённый путь'),
'a1_self':('я сейчас проверю уплотнение','сначала я проверю уплотнение','я теперь проверю уплотнение'),
'a2_self':('вместо этого я проверю защёлку','затем я проверю защёлку','я теперь проверю защёлку'),
'b2_other':('затем ты закрепишь крышку','после этого ты закрепишь крышку','затем ты закрепишь крышку вместе со мной'),
'b2_self':('затем я закреплю крышку','после этого я закреплю крышку','я теперь закреплю крышку')},
'ja':{
'b1_other':('あなたは今、安全な経路を選ぶ','あなたはまず安全な経路を選ぶ','あなたは私と一緒に安全な経路を選ぶ'),
'b1_self':('私は今、安全な経路を選ぶ','私はまず安全な経路を選ぶ','私は次に安全な経路を選ぶ'),
'a1_self':('私は今、シールを点検する','私はまずシールを点検する','私は一緒にシールを点検する'),
'a2_self':('私は代わりにラッチを点検する','私は次にラッチを点検する','私は一緒にラッチを点検する'),
'b2_other':('次にあなたがカバーを固定する','その後あなたがカバーを固定する','次にあなたが一緒にカバーを固定する'),
'b2_self':('次に私がカバーを固定する','その後私がカバーを固定する','私は今度カバーを固定する')},
'zh':{
'b1_other':('你现在走安全路线','你先走安全路线','你现在先走安全路线'),
'b1_self':('我现在走安全路线','我先走安全路线','我现在先走安全路线'),
'a1_self':('我现在检查密封件','我先检查密封件','我现在先检查密封件'),
'a2_self':('我改为检查锁扣','我接着检查锁扣','我现在先检查锁扣'),
'b2_other':('然后你固定盖板','之后你固定盖板','然后你和我一起固定盖板'),
'b2_self':('然后我固定盖板','之后我固定盖板','我现在先固定盖板')},
'mixed':{
'b1_other':('you nimmst the sheltered Weg now','you nimmst first the sheltered Weg','you nimmst together the sheltered Weg'),
'b1_self':('I nehme the sheltered Weg now','I nehme first the sheltered Weg','I nehme together the sheltered Weg'),
'a1_self':('I prüfe the seal now','I prüfe first the seal','I prüfe the seal together'),
'a2_self':('I teste the latch instead','I teste next the latch','I teste the latch together'),
'b2_other':('then you sicherst the cover','after that you sicherst the cover','then you sicherst the cover together'),
'b2_self':('then I sichere the cover','after that I sichere the cover','I sichere the cover together')},
'denglish':{
'b1_other':('du nimmst jetzt den safe Weg, bro','du nimmst zuerst den safe Weg, bro','du nimmst den safe Weg mit mir, safe'),
'b1_self':('ich nehme jetzt den safe Weg, bro','ich nehme zuerst den safe Weg, bro','ich nehme den safe Weg mit dir, safe'),
'a1_self':('ich checke jetzt die Dichtung, bro','ich checke zuerst die Dichtung, bro','ich checke die Dichtung mit dir, safe'),
'a2_self':('ich teste stattdessen den Latch, bro','ich teste als Nächstes den Latch, bro','ich teste den Latch mit dir, safe'),
'b2_other':('danach securest du die Cover, bro','anschließend securest du die Cover, bro','danach securest du die Cover mit mir, safe'),
'b2_self':('danach secure ich die Cover, bro','anschließend secure ich die Cover, bro','danach secure ich die Cover, safe')}}

ACTION_WORDS={
'en':{B1:'take the sheltered route',B2:'secure the cover',A1:'inspect the seal',A2:'test the latch'},
'de':{B1:'den geschützten Weg nehmen',B2:'die Abdeckung sichern',A1:'die Dichtung prüfen',A2:'die Verriegelung testen'},
'ru':{B1:'выбрать защищённый путь',B2:'закрепить крышку',A1:'проверить уплотнение',A2:'проверить защёлку'},
'ja':{B1:'安全な経路を選ぶ',B2:'カバーを固定する',A1:'シールを点検する',A2:'ラッチを点検する'},
'zh':{B1:'走安全路线',B2:'固定盖板',A1:'检查密封件',A2:'检查锁扣'},
'mixed':{B1:'take den sheltered Weg',B2:'secure die cover',A1:'inspect die Dichtung',A2:'test den latch'},
'denglish':{B1:'den safe Weg nehmen',B2:'die Cover securen',A1:'die Dichtung checken',A2:'den Latch testen'}}
QUESTIONS={
'en':('Should you take the sheltered route or secure the cover?','Should you secure the cover or inspect the seal?','Should you inspect the seal or test the latch?'),
'de':('Sollst du den geschützten Weg nehmen oder die Abdeckung sichern?','Sollst du die Abdeckung sichern oder die Dichtung prüfen?','Sollst du die Dichtung prüfen oder die Verriegelung testen?'),
'ru':('Тебе выбрать защищённый путь или закрепить крышку?','Тебе закрепить крышку или проверить уплотнение?','Тебе проверить уплотнение или проверить защёлку?'),
'ja':('安全な経路を選ぶ、それともカバーを固定する？','カバーを固定する、それともシールを点検する？','シールを点検する、それともラッチを点検する？'),
'zh':('你应该走安全路线还是固定盖板？','你应该固定盖板还是检查密封件？','你应该检查密封件还是检查锁扣？'),
'mixed':('Should you take den sheltered Weg or secure die cover?','Should you secure die cover or inspect die Dichtung?','Should you inspect die Dichtung or test den latch?'),
'denglish':('Sollst du den safe Weg nehmen oder die Cover securen?','Sollst du die Cover securen oder die Dichtung checken?','Sollst du die Dichtung checken oder den Latch testen?')}

def _task(i):base=0xF900+int(i)*0x20;return base+1,base+2,base+3

def build(start):
 from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
 rows=[]
 def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)));return rows[-1].sequence
 names=tuple(ROLE_FORMS)
 # Acute arousal from the prior experimental episode decays; slow relationship/control history remains.
 add('quiet',0,(24,))
 for idx,name in enumerate(names):
  base=0xFA00+idx*0x100;forms=ROLE_FORMS[name]
  for witness in range(4):
   speaker=base+witness;other=base+0x40+witness;variant=witness%2
   for key,action,actor_self in (('b1_other',B1,False),('b1_self',B1,True),('a1_self',A1,True),('a2_self',A2,True),('b2_other',B2,False),('b2_self',B2,True)):
    add('raw_speech_contact',speaker,tuple(forms[key][variant].encode('utf-8')))
    add('observed_joint_action',speaker if actor_self else other,(speaker,action))
  # Ground action names and two non-focal binary inquiry examples in the embodied language.
  for action,text in ACTION_WORDS[name].items():
   for source in (base+2,base+3):
    add('embodied_language_scene',source,(100,action));add('embodied_language_surface',source,tuple(text.encode('utf-8')))
  for atoms,q in (((B1,B2),QUESTIONS[name][0]),((B2,A1),QUESTIONS[name][1])):
   for source in (base+2,base+3):
    add('embodied_language_scene',source,(INQUIRY_CONTEXT,*atoms));add('embodied_language_surface',source,tuple(q.encode('utf-8')))
 add('checkpoint_mark',0,('argument_joint_role_and_inquiry_grounding',))

 for idx,name in enumerate(names):
  base=0xFA00+idx*0x100;partner=base+3;channel=0xD00+idx;state,mid,goal=_task(idx)
  add('embodied_partner_context',partner,(channel,partner));add('embodied_context',partner,(state,goal,(A1,A2,B1,B2)))
  r1=add('embodied_shared_reason',partner,(R1,Q//2));r2=add('embodied_shared_reason',partner,(R2,Q//2));r3=add('embodied_shared_reason',partner,(R3,Q//2))
  forms=ROLE_FORMS[name];clauses=tuple(tuple(forms[key][2].encode('utf-8')) for key in ('b1_other','a1_self','b2_other'))
  add('embodied_joint_language_instruction',partner,(clauses,((r1,),(r2,),(r3,))))
  add('embodied_joint_public_opportunity',partner,())
  add('embodied_episode_step',0xFC00+idx,(state,goal,(A1,B1),((A1,mid,1),(B1,mid,1))))
  add('authenticated_utterance',partner,tuple(forms['a2_self'][2].encode('utf-8')))
  add('embodied_joint_public_opportunity',partner,())
  add('authenticated_utterance',partner,tuple(forms['a2_self'][2].encode('utf-8')))
  add('embodied_joint_partner_observation',partner,(A2,1))
  add('embodied_joint_public_opportunity',partner,())
  add('embodied_episode_step',0xFD00+idx,(mid,goal,(A2,B2),((A2,goal,1),(B2,goal,1))))
  add('checkpoint_mark',0,(f'argument_joint_plan_{name}',))
 add('checkpoint_mark',0,('argument_joint_plan_seven_register',))
 return tuple(rows)
