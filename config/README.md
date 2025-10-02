The toolkit configuration is very flexible. 

- Mandatory file is `config/app_conf.yaml`, that can load and merge other YAML files. 
- Under the hood, we use OmegaConf, that notably allows  variable values interpolation. Have a look at it: https://omegaconf.readthedocs.io/en/2.3_branch/grammar.html
- Here we have a basic configuration for test and exemple purpose.  You might reoganized it!
- The configuraration can be overwriten by setting the variable `default_configuration` in `config/app_conf.yaml`.  It's a good practice to allow configuration selection through an environment variale (such as `BLUEPRINT_CONFIG` here). When configured, first load the files defined in 'merge', then merge with a dict whose key is the `default_configuration` value.