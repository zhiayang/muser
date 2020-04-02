// config.cpp
// Copyright (c) 2019, zhiayang
// Licensed under the Apache License Version 2.0.

#include "defs.h"

#include "picojson.h"

namespace pj = picojson;

namespace config
{
	static std::fs::path getDefaultConfigPath()
	{
		auto home = std::fs::path(util::getEnvironmentVar("HOME"));
		if(!home.empty())
		{
			auto x = home / ".config" / "mkvtaginator" / "config.json";
			if(std::fs::exists(x))
				return x;
		}

		if(std::fs::exists("mkvtaginator-config.json"))
			return std::fs::path("mkvtaginator-config.json");

		if(std::fs::exists(".mkvtaginator-config.json"))
			return std::fs::path(".mkvtaginator-config.json");

		return "";
	}

	template <typename... Args>
	void error(const std::string& fmt, Args&&... args)
	{
		util::error(fmt, args...);
	}

	void readConfig()
	{
		// if there's a manual one, use that.
		std::fs::path path;
		if(auto cp = getConfigPath(); !cp.empty())
		{
			path = cp;
			if(!std::fs::exists(path))
			{
				// ...
				util::error("specified configuration file '%s' does not exist", cp);
				return;
			}
		}

		if(auto cp = getDefaultConfigPath(); !cp.empty())
		{
			path = cp;

			// read it.
			uint8_t* buf = 0; size_t sz = 0;
			std::tie(buf, sz) = util::readEntireFile(path.string());
			if(!buf || sz == 0)
			{
				error("failed to read file");
				return;
			}

			util::log("reading config file '%s'", path.string());


			pj::value config;

			auto begin = buf;
			auto end = buf + sz;
			std::string err;
			pj::parse(config, begin, end, &err);
			if(!err.empty())
			{
				error("%s", err);
				return;
			}

			// the top-level object should be "options".
			if(auto options = config.get("options"); !options.is<pj::null>())
			{

				auto opts = options.get<pj::object>();

				auto get_string = [&opts](const std::string& key, const std::string& def) -> std::string {
					if(auto it = opts.find(key); it != opts.end())
					{
						if(it->second.is<std::string>())
							return it->second.get<std::string>();

						else
							error("expected string value for '%s'", key);
					}

					return def;
				};

				auto get_array = [&opts](const std::string& key) -> std::vector<pj::value> {
					if(auto it = opts.find(key); it != opts.end())
					{
						if(it->second.is<pj::array>())
							return it->second.get<pj::array>();

						else
							error("expected array value for '%s'", key);
					}

					return { };
				};

				auto get_bool = [&opts](const std::string& key, bool def) -> bool {
					if(auto it = opts.find(key); it != opts.end())
					{
						if(it->second.is<bool>())
							return it->second.get<bool>();

						else
							error("expected boolean value for '%s'", key);
					}

					return def;
				};
			}
			else
			{
				error("no top-level 'options' object");
			}

			delete[] buf;
		}

		// it's ok not to have one.
	}
















	static std::string configPath;


	std::string getConfigPath()             { return configPath; }







	void setConfigPath(const std::string& x)
	{
		// this one is special. once we set it, we wanna re-read the config.
		configPath = x;
		readConfig();
	}
}









