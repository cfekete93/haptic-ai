from enum import Enum
import logging

import intent_classifier as ic
import json
import re

import coin_manager as cm


class Service:
    class Type(Enum):
        INVALID = 0
        VIEW    = 1
        SELL    = 2
        BUY     = 3

    def __init__(self, service_type: Type, complete_query: bool, **entities):
        self.service_type = service_type
        self.complete_query = complete_query
        self._entities = entities

    def __getitem__(self, item: str):
        return self._entities.get(item, None)

    def __iter__(self):
        return iter(self._entities)

    def __str__(self):
        if self.service_type == Service.Type.INVALID:
            return "Invalid service request!"
        return f'Service request of type {self.service_type.name} ' \
               f'that is {"complete" if self.complete_query else "incomplete"} ' \
               f'with entities: {self._entities}'

    @property
    def type(self) -> Type:
        return self._service_type

    @type.setter
    def type(self, service_type: Type):
        self._service_type = service_type
    
    @property
    def complete_query(self) -> bool:
        return self._complete_query
    
    @complete_query.setter
    def complete_query(self, complete_query: bool):
        self._complete_query = complete_query

    def to_dict(self):
        return {'service_type': self.service_type.name,
                'complete_query': self.complete_query,
                'entities': self._entities}

    def to_json(self):
        request = {'service_type': self.service_type.name,
                   'complete_query': self.complete_query,
                   'entities': self._entities}
        return json.dumps(request, indent=4)


class ServiceClassifier:
    def __init__(self, classifier: ic.IntentClassifier):
        self._intent_classifier = classifier

    def parse_request(self, request: str) -> Service | None:
        request = request.strip().lower()
        if len(request) <= 0:
            return None
        service_type = self._determine_service_type(request)
        return self._determine_service(request, service_type)

    @classmethod
    def load_intent_classifier(cls,
                               model_save_path: str = 'models/intents.h5',
                               classes_save_path: str = 'utils/classes.pkl',
                               tokenizer_save_path: str = 'utils/tokenizer.pkl',
                               label_encoder_save_path: str = 'utils/label_encoder.pkl'):
        model, classes, tokenizer, label_encoder = ic.load_intent_classifier(model_save_path,
                                                                             classes_save_path,
                                                                             tokenizer_save_path,
                                                                             label_encoder_save_path)
        return cls(ic.IntentClassifier(classes, model, tokenizer, label_encoder))

    def _determine_service_type(self, request) -> Service.Type:
        intent = self._get_intent(request)
        service_type = self._get_service_type(intent)
        if service_type is Service.Type.INVALID:
            intent = self._intent_fallback(request, intent)
            service_type = self._get_service_type(intent)
        return service_type

    def _get_intent(self, request: str) -> str | None:
        intent = self._intent_classifier.get_intent(request)
        if intent == 'order' or intent == 'view' or intent == 'buy' or intent == 'sell':
            return intent
        return None

    def _intent_fallback(self, request: str, response: str | None) -> str | None:
        matches, intent = 0, None

        check_buy = True
        check_sell = True
        check_view = True

        if response == 'order':
            check_view = True

        if check_buy and any(word in request.split() for word in ServiceClassifier._lookup_sets['buy']):
            matches += 1
            intent = 'buy'
        if check_sell and any(word in request.split() for word in ServiceClassifier._lookup_sets['sell']):
            matches += 1
            intent = 'sell'
        if check_view and any(word in request.split() for word in ServiceClassifier._lookup_sets['view']):
            matches += 1
            intent = 'view'

        if matches == 1:
            return intent
        return None

    def _get_service_type(self, intent: str) -> Service.Type:
        if intent == 'buy':
            return Service.Type.BUY
        if intent == 'sell':
            return Service.Type.SELL
        if intent == 'view':
            return Service.Type.VIEW
        return Service.Type.INVALID

    def _determine_service(self, request: str, service_type: Service.Type) -> Service | None:
        if service_type == Service.Type.INVALID:
            return Service(service_type, False)
        complete_query, entities = self._get_entities(request, service_type)
        if not complete_query:  # If met, either an incomplete user request or a bad look-up by the NER
            complete_query, entities = self._entity_fallback(request, service_type, entities)
        return Service(service_type, complete_query, **entities)

    def _get_entities(self, request: str, service_type: Service.Type) -> tuple[bool, dict[str, str]]:
        logging.warning("_get_entities called but is not fully implemented!")
        return False, {'message': 'NER check has been disabled!'}

    def _entity_fallback(self,
                         request: str,
                         service_type: Service.Type,
                         entities: dict[str, str]) -> tuple[bool, dict[str, str]]:
        if service_type is Service.Type.INVALID:
            return False, entities
        additional_entities = dict()
        # Find coin type
        if service_type == Service.Type.VIEW \
        or service_type == Service.Type.BUY \
        or service_type == Service.Type.SELL:
            for coin in cm.coins:
                search = [coin.get('name', '').lower(), coin.get('code', '').lower()]
                if search[0] == '' or search[1] == '':
                    continue
                if any(word in request.split() for word in search):
                    additional_entities['name'] = search[0]
                    break
        # Find quantity to buy/sell
        if service_type == Service.Type.BUY \
        or service_type == Service.Type.SELL:
            numbers = re.findall(r'\b\d+\b', request)
            if len(numbers) == 1:
                additional_entities['quantity'] = numbers[0]
            elif len(numbers) > 1:
                additional_entities['error'] = ['quantity', 'invalid: too many number received']  # TODO [Chris]: Make this append to an array named 'error'
        # Merge entity dicts
        # entities = {**additional_entities, **entities}  # Pre-python 3.9
        entities = additional_entities | entities  # Any overlapping keys will always be from second dict (entities)
        # Perform completion check
        complete_query = False
        match service_type:
            case Service.Type.VIEW:
                if entities.get('name', None) is not None:
                    complete_query = True
            case Service.Type.BUY:
                if entities.get('name', None) is not None \
                and entities.get('quantity', None) is not None:
                    complete_query = True
            case Service.Type.SELL:
                if entities.get('name', None) is not None \
                and entities.get('quantity', None) is not None:
                    complete_query = True
            case _:
                complete_query = False
        return complete_query, entities

    _lookup_sets = {
        'buy': {'buy', 'purchase', 'acquire', 'requisition'},
        'sell': {'sell', 'offload', 'cash-in'},
        'view': {'how much', 'what is', 'what', 'look up', 'lookup', 'look', 'tell me', 'value', 'price', 'see', 'show'}
    }


if __name__ != '__main__':
    service_classifier = ServiceClassifier.load_intent_classifier()
