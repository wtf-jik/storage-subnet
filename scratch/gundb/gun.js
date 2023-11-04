
const GUN = require('gun');

var gun = Gun(['http://localhost:8765/gun', 'https://gun-manhattan.herokuapp.com/gun']);

gun.get('test').put({hello: 'world'});