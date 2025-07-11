import express from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { createClient } from "redis";

const app = express();

app.use(express.json());      // if needed
app.use(express.urlencoded());

const PORT = process.env.BACKEND_PORT || 3000;
const HOST = process.env.BACKEND_HOST || '127.0.0.1';

console.log("Running server on " + HOST + ":" + PORT);

async function redis_client(host: string, port: number) {
  return await createClient({
    url: `redis://${host}:${port}`
  }).on("error", (err) => console.log("Redis Client Error", err))
  .connect();
}

const port = 3000;


app.get('/api/schema/:module/:classname', (req, res) => {
  const { module, classname } = req.params;
  const python = spawn('python3', ["-c", "import json; import "+ module + "; print(json.dumps(" + module + "." + classname + ".to_json_schema()))"]);

  let output = '';

  python.stdout.on('data', (data) => {
    output += data.toString();
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).send('Python script failed');
    }
    try {
      const result = JSON.parse(output); // Expecting Python to return valid JSON
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: 'Invalid JSON from Python script', raw: output });
    }
  });

});

app.get('/api/tractions', (req, res) => {
  const python = spawn('python3', ["-m", "pytractions.cli", "catalog"]);

  let output = '';

  python.stdout.on('data', (data) => {
    output += data.toString();
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).send('Python script failed');
    }
    try {
      const result = JSON.parse(output); // Expecting Python to return valid JSON
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: 'Invalid JSON from Python script', raw: output });
    }
  });
});

app.post('/api/run/:id/:module/:group/:classname', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const { id, module, group, classname } = req.params;

  const redisc = await redis_client('localhost', 6379);

  const model_data = {
    "classname": classname,
    "module": module,
    "group": group,
    "model": req.body,
    "id": id,
  }
  console.log("Storing traction model:", model_data);

  redisc.set('traction-model-' + id, JSON.stringify(model_data));

  const python = spawn('python3',
                       ["-m", "pytractions.cli", "local_run", "--io-type", "JSON",
                         "--run-id", 'traction-logs-' + id,
                         "--observer", "none",
                         "--logger-handler", "redis",
                         "--logger-handler-redis-settings", '{"redis_url":"localhost","port":6379}',
                         module + ":" + classname]);
  //console.log("REQ", req);
  //console.log("BODY", req.body);
  delete req.body.user_description;
  console.log("Running traction:", module, classname);
  python.stdin.write(JSON.stringify(req.body));
  python.stdin.end();

  let output = '';

  python.stdout.on('data', (data) => {
    const _data = data.toString().replace("\n", "");
    console.log(`STDOUT: ${_data}`);
    res.write(data);
    //output += data.toString();
  });

  python.stderr.on('data', (data) => {
    const _data = data.toString().replace("\n", "");
    res.write(data);
    console.error(`STDERR: ${_data}`);
  });

  python.on('close', async (code) => {
    await redisc.xAdd('traction-logs-' + id, '*', {log: "=======================================\n", "level": "20"});
    res.write(`=======================================\n`);
    await redisc.xAdd('traction-logs-' + id, '*', {log: `Python script exited with code ${code}\n`, "level": "20"});
    res.write(`Python script exited with code ${code}\n`);
    await redisc.xAdd('traction-logs-' + id, '*', {log: "=======================================\n", "level": "20"});
    res.write(`=======================================\n`);
    res.end();
    console.log(`Python script exited with code ${code}`);
    return // res.status(500).send('Python script failed');
  });
});


app.get('/api/archive/', async (req, res) => {
  const redisc = await redis_client('localhost', 6379);

  const archives: {uid: string, user_desc: string}[] = [];
  const keys: string[] = await redisc.keys('traction-model-*');
  const values: string[] = (await redisc.mGet(keys) as string[]);
  values.forEach((value, index) => {
      const parsed = JSON.parse((value as string));
      archives.push({uid: parsed.id, user_desc: parsed.model.user_description});
  });
  res.json(archives);
})


app.get('/api/model/:id', async (req, res) => {
  const { id } = req.params;
  const redisc = await createClient({
    url: 'redis://localhost:6379'
  }).on("error", (err) => console.log("Redis Client Error", err))
  .connect();
  redisc.get('traction-model-' + id).then((redis_data: string | null) => {
    if (redis_data) {
      const { classname, module, group, model } = JSON.parse(redis_data);
      console.log("Loading schema for :", { classname, module});
      console.log("model", model);

      const python = spawn('python3', ["-c", "import json; import "+ module + "; print(json.dumps(" + module + "." + classname + ".to_json_schema()))"]);
      var schema: string = "";
      python.stdout.on('data', (data) => {
        schema += data.toString().replace(/^\s+|\s+$/g, '');
      })
      python.stderr.on('data', (data) => {
        console.error(`STDERR: ${data}`);
      });
      python.on('close', (code) => {
        if (code !== 0) {
          console.error(`Python script exited with code ${code}`);
          return res.status(500).send('Python script failed');
          return;
        }
        try {
          res.send({model: model, group: group, schema: JSON.parse(schema), classname: classname});
        } catch (e) {
          res.status(500).json({ error: 'Invalid JSON schema', raw: schema });
        }
      })
    } else {
      res.status(404).send('Model not found');
    }
  })
});

app.get('/api/watch/:stream_id', async (req, res) => {
  const redisc = await redis_client('localhost', 6379);

  const { stream_id } = req.params;
  const { last_id = '0-0', count = 10 } = req.query;

  try {
    const entries = await redisc.xRange(stream_id, (last_id as string), '+', {
      COUNT: parseInt((count as string))
    });

    // Format nicely for frontend
    // const formatted = entries.map((val: { id: string; message: { [x: string]: string; }; }, index: number) => {
    //
    // });

    res.json({ messages: entries });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to read from stream' });
  }
})


app.delete('/api/delete/:_id', async (req, res) => {
  const redisc = await redis_client('localhost', 6379);
  const { _id } = req.params;

  console.log("Delete traction with ID:", _id);
  await redisc.del('traction-model-' + _id);
  redisc.del('traction-logs-' + _id).then((result) => {
    if (result === 1) {
      res.status(200).send('Deleted successfully');
    } else {
      res.status(404).send('Not found');
    }
  });
})

app.listen(Number(PORT), HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
});

