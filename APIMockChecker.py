import json
import traceback
from jsonschema import validate
from faker import Faker
import random
import sys
import logging
import re
from urllib.parse import urlparse, parse_qs
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

fake = Faker()

class BankAPIMockChecker:
    def __init__(self, openapi_spec_path):
        with open(openapi_spec_path, "r") as f:
            self.openapi_spec = json.load(f)
        self.max_recursion_depth = 10

    def mock_execute_request(self, api_name, url, method, endpoint, params=None):
        logger.debug(f"Starting mock_execute_request for {method} {endpoint}")
        try:
            if not self._validate_url(url):
                raise ValueError(f"Invalid URL: {url}")

            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)

            path_params = self._extract_path_params(endpoint)
            
            formatted_endpoint = self._format_endpoint(endpoint)

            if formatted_endpoint not in self.openapi_spec["paths"]:
                raise ValueError(f"Endpoint {formatted_endpoint} not found in OpenAPI spec")

            if method.lower() not in self.openapi_spec["paths"][formatted_endpoint]:
                raise ValueError(f"Method {method} not found for endpoint {formatted_endpoint}")

            if not url.startswith(
                f"{parsed_url.scheme}://{parsed_url.netloc}{self._get_base_url()}"
            ):
                raise ValueError(
                    f"URL does not match the base URL defined in the OpenAPI spec: {url}"
                )

            endpoint_spec = self.openapi_spec["paths"][formatted_endpoint][method.lower()]

            self._validate_params(params, query_params, path_params, endpoint_spec.get("parameters", []))

            response_spec = endpoint_spec["responses"]
            success_code = next(
                (code for code in response_spec.keys() if code.startswith("2")), "200"
            )

            if success_code == "204":
                mock_response = None
            elif "content" in response_spec[success_code]:
                response_schema = response_spec[success_code]["content"]["application/json"]["schema"]
                mock_response = self._generate_mock_response(response_schema)
            else:
                mock_response = {}

            if mock_response is not None:
                try:
                    response_schema = response_spec[success_code].get("content", {}).get("application/json", {}).get("schema", {})
                    validate(instance=mock_response, schema=response_schema)
                    logger.info("Mock response successfully validated against the schema.")
                except Exception as e:
                    logger.error(f"Schema validation error: {e}")

            logger.info(f"Mock execution completed with status code: {success_code}")
            return {"status_code": int(success_code), "data": mock_response}
        
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return {"status_code": 400, "data": {"error": str(ve)}}
        except Exception as e:
            logger.error(f"Error in mock_execute_request: {str(e)}")
            logger.error(traceback.format_exc()) 
            return {"status_code": 500, "data": {"error": "Internal server error"}}

    def _validate_url(self, url):
        parsed_url = urlparse(url)
        base_url = self._get_base_url()
        return base_url is None or parsed_url.path.startswith(base_url)

    def _get_base_url(self):
        if "servers" in self.openapi_spec:
            return urlparse(self.openapi_spec["servers"][0]["url"]).path
        elif "basePath" in self.openapi_spec:
            return self.openapi_spec["basePath"]
        else:
            logger.warning("No 'servers' or 'basePath' found in OpenAPI spec. URL validation will be skipped.")
            return None

    def _extract_path_params(self, endpoint):
        return re.findall(r'\{([^}]+)\}', endpoint)

    def _format_endpoint(self, endpoint):
        parts = endpoint.split('/')
        formatted_parts = []
        for part in parts:
            if part.startswith('{') and part.endswith('}'):
                formatted_parts.append(part)
            elif '{' in part and '}' in part:
                formatted_parts.append(part)
            elif '_' in part:
                formatted_parts.append(part)
            else:
                matched = False
                for spec_endpoint in self.openapi_spec["paths"].keys():
                    if part in spec_endpoint.split('/'):
                        formatted_parts.append(part)
                        matched = True
                        break
                if not matched:
                    formatted_parts.append(part)  
        return '/'.join(formatted_parts)

    def _validate_params(self, params, query_params, path_params, param_specs):
        resolved_param_specs = self._resolve_param_specs(param_specs)
        
        all_params = {**params} if params else {}
        all_params.update({k: v[0] if isinstance(v, list) else v for k, v in query_params.items()})
        
        for path_param in path_params:
            if path_param not in all_params:
                raise ValueError(f"Missing path parameter: {path_param}")

        expected_params = set(spec.get("name") for spec in resolved_param_specs if "name" in spec)
        expected_params.update(set(path_params))
        unexpected_params = set(all_params.keys()) - expected_params
        if unexpected_params:
            raise ValueError(f"Unexpected parameter(s): {', '.join(unexpected_params)}")

        for param_spec in resolved_param_specs:
            if "name" not in param_spec:
                logger.warning(f"Parameter specification missing 'name': {param_spec}")
                continue

            param_name = param_spec["name"]
            if param_name in all_params:
                logger.debug(f"Validating parameter: {param_name}")
                param_value = all_params[param_name]
                schema = self._resolve_ref(param_spec.get("schema", {}))

                self._validate_param_type(param_name, param_value, schema.get("type"))

                self._validate_enum(param_name, param_value, schema.get("enum"))

                self._validate_pattern(param_name, param_value, schema.get("pattern"))

            elif param_spec.get("required", False) and param_name not in path_params:
                raise ValueError(f"Missing required parameter: {param_name}")

    def _resolve_param_specs(self, param_specs):
        resolved_specs = []
        for param_spec in param_specs:
            if "$ref" in param_spec:
                resolved_spec = self._resolve_ref(param_spec)
                resolved_specs.append(resolved_spec)
            else:
                resolved_specs.append(param_spec)
        return resolved_specs

    def _resolve_ref(self, schema):
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")[1:]
            resolved_schema = self.openapi_spec
            for path in ref_path:
                resolved_schema = resolved_schema.get(path)
                if resolved_schema is None:
                    logger.error(f"Failed to resolve $ref: {schema['$ref']}")
                    return schema
            return resolved_schema
        return schema

    def _validate_param_type(self, param_name, param_value, expected_type):
        type_checks = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int) or (isinstance(x, str) and x.isdigit()),
            "number": lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').isdigit()),
            "boolean": lambda x: isinstance(x, bool) or (isinstance(x, str) and x.lower() in ['true', 'false'])
        }
        if expected_type in type_checks and not type_checks[expected_type](param_value):
            raise ValueError(f"Invalid type for parameter '{param_name}'. Expected {expected_type}.")

    def _validate_enum(self, param_name, param_value, enum_values):
        if enum_values and param_value not in enum_values:
            raise ValueError(f"Invalid value for parameter '{param_name}'. Must be one of: {enum_values}")

    def _validate_pattern(self, param_name, param_value, pattern):
        if pattern and not re.match(pattern, str(param_value)):
            raise ValueError(f"Invalid format for parameter '{param_name}'. Must match pattern: {pattern}")

    def _generate_mock_response(self, schema, depth=0):
        logger.debug(f"Generating mock response for schema: {schema}")
        if depth > self.max_recursion_depth:
            return "Max recursion depth exceeded"

        if "type" not in schema:
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")
                ref_schema = self.openapi_spec
                for path in ref_path[1:]:
                    ref_schema = ref_schema[path]
                return self._generate_mock_response(ref_schema, depth + 1)
            return {}

        if schema["type"] == "object":
            response = {}
            if "properties" in schema:
                for prop, prop_schema in schema["properties"].items():
                    response[prop] = self._generate_mock_response(
                        prop_schema, depth + 1
                    )
            return response
        elif schema["type"] == "array":
            items_schema = schema.get("items", {})
            return [
                self._generate_mock_response(items_schema, depth + 1)
                for _ in range(random.randint(1, 3))
            ]
        elif schema["type"] == "string":
            if "enum" in schema:
                return random.choice(schema["enum"])
            elif "format" in schema:
                if schema["format"] == "date-time":
                    return fake.iso8601()
                elif schema["format"] == "date":
                    return fake.date_this_decade().isoformat()
            return fake.word()
        elif schema["type"] == "number":
            return round(random.uniform(0, 100), 2)
        elif schema["type"] == "integer":
            return random.randint(0, 100)
        elif schema["type"] == "boolean":
            return random.choice([True, False])
        else:
            logger.warning(f"Unexpected schema type: {schema['type']}")
            return {}

def read_output_dataset(file_path):
    requests = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for answer in data.get('answers', []):
                request = {
                    "api_name": answer.get('api_name', ''),
                    "url": answer.get('url', ''),
                    "method": answer.get('method', ''),
                    "endpoint": answer.get('endpoint', ''),
                    "params": answer.get('params', {})
                }
                requests.append(request)
    return requests

def read_openapi_specs(folder_path):
    specs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    spec = json.load(file)
                    api_name = spec.get('info', {}).get('title', '').replace(' ', '_')
                    specs[api_name] = file_path
            except UnicodeDecodeError:
                logger.warning(f"Unable to read {file_path} with UTF-8 encoding. Trying with ISO-8859-1.")
                try:
                    with open(file_path, 'r', encoding='iso-8859-1') as file:
                        spec = json.load(file)
                        api_name = spec.get('info', {}).get('title', '').replace(' ', '_')
                        specs[api_name] = file_path
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {str(e)}")
    return specs

def match_and_execute_requests(requests, specs):
    results = []
    for request in requests:
        api_name = request['api_name']
        if api_name in specs:
            spec_path = specs[api_name]
            checker = BankAPIMockChecker(spec_path)
            response = checker.mock_execute_request(
                api_name,
                request['url'],
                request['method'],
                request['endpoint'],
                request['params']
            )
            results.append({
                'request': request,
                'response': response,
                'spec_file': spec_path
            })
        else:
            results.append({
                'request': request,
                'error': f"No matching OpenAPI spec found for {api_name}",
                'spec_file': None
            })
    return results
def main():
    try:
        logger.info("Starting main function")

        output_dataset_path = 'APIs/output_dataset.jsonl'
        apis_folder_path = 'APIs'

        requests = read_output_dataset(output_dataset_path)
        specs = read_openapi_specs(apis_folder_path)
        results = match_and_execute_requests(requests, specs)

        status_code_summary = {}

        for i, result in enumerate(results, start=1):
            logger.info(f"Result for request {i}:")
            logger.info(f"Request: {json.dumps(result['request'], indent=2)}")
            if 'response' in result:
                response_code = result['response']['status_code']
                status_code_summary[response_code] = status_code_summary.get(response_code, 0) + 1
                
                logger.info(f"Response status code: {response_code}")
                logger.info(f"Response data: {json.dumps(result['response']['data'], indent=2)}")
            else:
                logger.info(f"Error: {result['error']}")
            
            logger.info(f"Spec file: {result['spec_file']}")
            logger.info("---")

        logger.info("All requests processed")

        total_requests = len(results)
        successful_requests = sum(count for code, count in status_code_summary.items() if code < 400)  
        failed_requests = total_requests - successful_requests
        
        logger.info("Summary Report:")
        logger.info(f"Total requests: {total_requests}")
        logger.info(f"Successful requests: {successful_requests}")
        logger.info(f"Failed requests: {failed_requests}")

        for status_code, count in status_code_summary.items():
            logger.info(f"Status Code {status_code}: {count} occurrences")

    except Exception as e:
        logger.critical(f"Unexpected error in main function: {str(e)}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    sys.setrecursionlimit(100) 
    main()
