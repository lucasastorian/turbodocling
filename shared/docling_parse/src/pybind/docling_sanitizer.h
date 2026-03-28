//-*-C++-*-

#ifndef PYBIND_PDF_SANITIZER_H
#define PYBIND_PDF_SANITIZER_H

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <v2.h>

namespace docling
{
  namespace sanitizer_profile
  {
    inline bool enabled()
    {
      auto const* env = std::getenv("TD_SAN_PROFILE");
      return env != nullptr && std::string(env) == "1";
    }

    inline double elapsed_ms(std::chrono::steady_clock::time_point start,
			     std::chrono::steady_clock::time_point end)
    {
      return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    }

    inline void print(const char* phase,
		      std::size_t input_cells,
		      std::size_t output_cells,
		      double copy_ms,
		      double filter_ms,
		      double sanitize_ms,
		      double emit_ms,
		      double total_ms)
    {
      if(not enabled())
	{
	  return;
	}

      std::fprintf(stderr,
		   "[docling_parse] sanitizer_%s: in=%zu out=%zu | copy=%.0fms filter=%.0fms sanitize=%.0fms emit=%.0fms total=%.0fms\n",
		   phase,
		   input_cells,
		   output_cells,
		   copy_ms,
		   filter_ms,
		   sanitize_ms,
		   emit_ms,
		   total_ms);
    }
  }

  class docling_sanitizer: public docling_resources
  {
  public:

    docling_sanitizer();

    docling_sanitizer(std::string level);

    void set_loglevel(int level=0);
    void set_loglevel_with_label(std::string level="error");

    bool set_char_cells(nlohmann::json& data);

    nlohmann::json to_records(std::string label);

    nlohmann::json create_word_cells(double horizontal_cell_tolerance=1.00,
				     bool enforce_same_font=true,
				     double space_width_factor_for_merge=0.05);
    nlohmann::json create_word_cells_table(double horizontal_cell_tolerance=1.00,
					   bool enforce_same_font=true,
					   double space_width_factor_for_merge=0.05);
    nlohmann::json create_word_cells_table_compact(double horizontal_cell_tolerance=1.00,
						   bool enforce_same_font=true,
						   double space_width_factor_for_merge=0.05);

    nlohmann::json create_line_cells(double horizontal_cell_tolerance=1.00,
				     bool enforce_same_font=true,
				     double space_width_factor_for_merge=1.00,
				     double space_width_factor_for_merge_with_space=0.33);
    nlohmann::json create_line_cells_table(double horizontal_cell_tolerance=1.00,
					   bool enforce_same_font=true,
					   double space_width_factor_for_merge=1.00,
					   double space_width_factor_for_merge_with_space=0.33);
    nlohmann::json create_line_cells_table_compact(double horizontal_cell_tolerance=1.00,
						   bool enforce_same_font=true,
						   double space_width_factor_for_merge=1.00,
						   double space_width_factor_for_merge_with_space=0.33);
    
  private:

    pdflib::pdf_sanitator<pdflib::PAGE_CELLS> cell_sanitizer;

    pdflib::pdf_resource<pdflib::PAGE_CELLS> char_cells;
    pdflib::pdf_resource<pdflib::PAGE_CELLS> word_cells;
    pdflib::pdf_resource<pdflib::PAGE_CELLS> line_cells;    
  };

  docling_sanitizer::docling_sanitizer():
    cell_sanitizer(),

    char_cells(),
    word_cells(),
    line_cells()    
  {}

  docling_sanitizer::docling_sanitizer(std::string level)
  {
    set_loglevel_with_label(level);
  }

  void docling_sanitizer::set_loglevel(int level)
  {
    if(level>=3)
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
      }
    else if(level==2)
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
      }
    else if(level==1)
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_ERROR;
      }
    else if(level==0)
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;
      }
    else
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_ERROR;
      }
  }

  void docling_sanitizer::set_loglevel_with_label(std::string level)
  {
    if(level=="info")
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
      }
    else if(level=="warning" or level=="warn")
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
      }
    else if(level=="error")
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_ERROR;
      }
    else if(level=="fatal")
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;
      }
    else
      {
        loguru::g_stderr_verbosity = loguru::Verbosity_ERROR;
      }
  }
  
  bool docling_sanitizer::set_char_cells(nlohmann::json& data)
  {
    LOG_S(INFO) << __FUNCTION__;
    
    if(data.is_array())
      {
	if(data.empty() or data.front().is_array())
	  {
	    char_cells.init_from(data);
	  }
	else
	  {
	    char_cells.clear();
	
	    for(auto& item:data)
	      {
		pdflib::pdf_resource<pdflib::PAGE_CELL> char_cell;

		{
		  char_cell.active = true;
	    
		  char_cell.r_x0 = item.at("rect").at("r_x0").get<double>();
		  char_cell.r_y0 = item.at("rect").at("r_y0").get<double>();

		  char_cell.r_x1 = item.at("rect").at("r_x1").get<double>();
		  char_cell.r_y1 = item.at("rect").at("r_y1").get<double>();

		  char_cell.r_x2 = item.at("rect").at("r_x2").get<double>();
		  char_cell.r_y2 = item.at("rect").at("r_y2").get<double>();

		  char_cell.r_x3 = item.at("rect").at("r_x3").get<double>();
		  char_cell.r_y3 = item.at("rect").at("r_y3").get<double>();

		  /*
		  char_cell.x0
		    = std::min(char_cell.r_x0, std::min(char_cell.r_x1, std::min(char_cell.r_x2, char_cell.r_x3)));
		  char_cell.y0
		    = std::min(char_cell.r_y0, std::min(char_cell.r_y1, std::min(char_cell.r_y2, char_cell.r_y3)));

		  char_cell.x1
		    = std::max(char_cell.r_x0, std::max(char_cell.r_x1, std::max(char_cell.r_x2, char_cell.r_x3)));
		  char_cell.y1
		  = std::max(char_cell.r_y0, std::max(char_cell.r_y1, std::max(char_cell.r_y2, char_cell.r_y3)));	   
		  */
	      
		  char_cell.text = item.at("text").get<std::string>();
		  //char_cell.orig = item.at("text").get<std::string>();
	      
		  char_cell.rendering_mode = item.at("rendering_mode").get<int>();

		  char_cell.font_name = item.at("font_name").get<std::string>();
		  char_cell.font_key = item.at("font_key").get<std::string>();

		  char_cell.left_to_right = item.at("left_to_right").get<bool>();
		  char_cell.widget = item.at("widget").get<bool>();	      
		}
		char_cells.push_back(char_cell);	    
	      }
	  }

	LOG_S(INFO) << "read " << char_cells.size() << " char-cells";
	
	return true;
      }
    else if(data.is_object())
      {
	if(data.contains("data"))
	  {
	    auto const has_header = data.contains("header") && data.at("header").is_array();
	    if(has_header)
	      {
		auto& header = data.at("header");
		auto& rows = data.at("data");
		std::map<std::string, int> indices;
		for(int i=0; i<header.size(); i++)
		  {
		    indices[header.at(i).get<std::string>()] = i;
		  }

		auto get_required_index = [&](std::string const& name) -> int {
		  auto itr = indices.find(name);
		  if(itr == indices.end())
		    {
		      std::stringstream ss;
		      ss << "missing required char-cell column `" << name << "`";
		      throw std::logic_error(ss.str());
		    }
		  return itr->second;
		};

		auto get_optional_string = [&](nlohmann::json& row, std::string const& name) -> std::string {
		  auto itr = indices.find(name);
		  return (itr == indices.end()) ? std::string("") : row.at(itr->second).get<std::string>();
		};

		auto get_optional_double = [&](nlohmann::json& row, std::string const& name, double fallback) -> double {
		  auto itr = indices.find(name);
		  return (itr == indices.end()) ? fallback : row.at(itr->second).get<double>();
		};

		int const rx0 = get_required_index("r_x0");
		int const ry0 = get_required_index("r_y0");
		int const rx1 = get_required_index("r_x1");
		int const ry1 = get_required_index("r_y1");
		int const rx2 = get_required_index("r_x2");
		int const ry2 = get_required_index("r_y2");
		int const rx3 = get_required_index("r_x3");
		int const ry3 = get_required_index("r_y3");
		int const text = get_required_index("text");
		int const rendering_mode = get_required_index("rendering-mode");
		int const font_key = get_required_index("font-key");
		int const font_name = get_required_index("font-name");
		int const widget = get_required_index("widget");
		int const left_to_right = get_required_index("left_to_right");

		char_cells.clear();
		for(auto& row:rows)
		  {
		    pdflib::pdf_resource<pdflib::PAGE_CELL> char_cell;
		    char_cell.active = true;

		    char_cell.r_x0 = row.at(rx0).get<double>();
		    char_cell.r_y0 = row.at(ry0).get<double>();
		    char_cell.r_x1 = row.at(rx1).get<double>();
		    char_cell.r_y1 = row.at(ry1).get<double>();
		    char_cell.r_x2 = row.at(rx2).get<double>();
		    char_cell.r_y2 = row.at(ry2).get<double>();
		    char_cell.r_x3 = row.at(rx3).get<double>();
		    char_cell.r_y3 = row.at(ry3).get<double>();

		    char_cell.x0 = get_optional_double(row, "x0", std::min(char_cell.r_x0, std::min(char_cell.r_x1, std::min(char_cell.r_x2, char_cell.r_x3))));
		    char_cell.y0 = get_optional_double(row, "y0", std::min(char_cell.r_y0, std::min(char_cell.r_y1, std::min(char_cell.r_y2, char_cell.r_y3))));
		    char_cell.x1 = get_optional_double(row, "x1", std::max(char_cell.r_x0, std::max(char_cell.r_x1, std::max(char_cell.r_x2, char_cell.r_x3))));
		    char_cell.y1 = get_optional_double(row, "y1", std::max(char_cell.r_y0, std::max(char_cell.r_y1, std::max(char_cell.r_y2, char_cell.r_y3))));

		    char_cell.text = row.at(text).get<std::string>();
		    char_cell.rendering_mode = row.at(rendering_mode).get<int>();
		    char_cell.space_width = get_optional_double(row, "space-width", 0.0);
		    char_cell.enc_name = get_optional_string(row, "encoding-name");
		    char_cell.font_enc = get_optional_string(row, "font-encoding");
		    char_cell.font_key = row.at(font_key).get<std::string>();
		    char_cell.font_name = row.at(font_name).get<std::string>();
		    char_cell.left_to_right = row.at(left_to_right).get<bool>();
		    char_cell.widget = row.at(widget).get<bool>();

		    char_cells.push_back(char_cell);
		  }
	      }
	    else
	      {
		auto& rows = data.at("data");
		char_cells.init_from(rows);
	      }
	  }
	else
	  {
	    char_cells.init_from(data);
	  }
	LOG_S(INFO) << "read " << char_cells.size() << " char-cells";
	
	return true;
      }
    else
      {
	LOG_S(ERROR) << "could not interprete data as char_cells: " << data.dump(2); 
      }

    return false;
  }

  nlohmann::json docling_sanitizer::to_records(std::string label)
  {
    LOG_S(INFO) << __FUNCTION__;

    nlohmann::json result = nlohmann::json::array({});
    
    pdflib::pdf_resource<pdflib::PAGE_CELLS>* cells = NULL;
    
    if(label=="char")
      {
	cells = &char_cells;
      }
    else if(label=="word")
      {
	cells = &word_cells;
      }
    else if(label=="line")
      {
	cells = &line_cells;
      }
    else
      {
	return result;
      }

    int order = 0;
    for(auto itr=cells->begin(); itr!=cells->end(); itr++)
      {
	pdflib::pdf_resource<pdflib::PAGE_CELL>& cell = *itr;

	if(not cell.active)
	  {
	    continue;
	  }
	
	nlohmann::json item = nlohmann::json::object({});

	{
	  nlohmann::json rect = nlohmann::json::object({});

	  rect["r_x0"] = cell.r_x0; rect["r_y0"] = cell.r_y0;
	  rect["r_x1"] = cell.r_x1; rect["r_y1"] = cell.r_y1;
	  rect["r_x2"] = cell.r_x2; rect["r_y2"] = cell.r_y2;
	  rect["r_x3"] = cell.r_x3; rect["r_y3"] = cell.r_y3;

	  item["index"] = (order++);
	  
	  item["rect"] = rect;

	  item["text"] = cell.text;
	  item["orig"] = cell.text;

	  item["font_key"] = cell.font_key;
	  item["font_name"] = cell.font_name;

	  item["rendering_mode"] = cell.rendering_mode;

	  item["widget"] = cell.widget;
	  item["left_to_right"] = cell.left_to_right;
	}

	result.push_back(item);
      }
    
    return result;
  }
  
  nlohmann::json docling_sanitizer::create_word_cells(double horizontal_cell_tolerance,
						      bool enforce_same_font,
						      double space_width_factor_for_merge)
  {
    LOG_S(INFO) << __FUNCTION__;

    // do a deep copy
    word_cells = char_cells;

    LOG_S(INFO) << "#-word cells: " << word_cells.size();
    
    auto new_end
      = std::remove_if(word_cells.begin(),
		       word_cells.end(),
		       [](auto& cell) {
			 return utils::string::is_space(cell.text);
		       });
    word_cells.erase(new_end, word_cells.end());

    LOG_S(INFO) << "#-word cells: " << word_cells.size();
    
    // > space_width_factor_for_merge, so nothing gets merged with a space
    double space_width_factor_for_merge_with_space = 2.0*space_width_factor_for_merge; 
    
    cell_sanitizer.sanitize_bbox(word_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);

    LOG_S(INFO) << "#-wordcells: " << word_cells.size();

    return to_records("word");
  }

  nlohmann::json docling_sanitizer::create_word_cells_table(double horizontal_cell_tolerance,
							    bool enforce_same_font,
							    double space_width_factor_for_merge)
  {
    LOG_S(INFO) << __FUNCTION__;

    auto total_start = std::chrono::steady_clock::now();

    word_cells = char_cells;
    auto after_copy = std::chrono::steady_clock::now();

    auto new_end
      = std::remove_if(word_cells.begin(),
		       word_cells.end(),
		       [](auto& cell) {
			 return utils::string::is_space(cell.text);
		       });
    word_cells.erase(new_end, word_cells.end());
    auto after_filter = std::chrono::steady_clock::now();

    double space_width_factor_for_merge_with_space = 2.0*space_width_factor_for_merge;

    cell_sanitizer.sanitize_bbox(word_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);
    auto after_sanitize = std::chrono::steady_clock::now();

    auto result = word_cells.get();
    auto after_emit = std::chrono::steady_clock::now();

    sanitizer_profile::print("word_table",
			     char_cells.size(),
			     word_cells.size(),
			     sanitizer_profile::elapsed_ms(total_start, after_copy),
			     sanitizer_profile::elapsed_ms(after_copy, after_filter),
			     sanitizer_profile::elapsed_ms(after_filter, after_sanitize),
			     sanitizer_profile::elapsed_ms(after_sanitize, after_emit),
			     sanitizer_profile::elapsed_ms(total_start, after_emit));

    return result;
  }

  nlohmann::json docling_sanitizer::create_word_cells_table_compact(double horizontal_cell_tolerance,
								    bool enforce_same_font,
								    double space_width_factor_for_merge)
  {
    LOG_S(INFO) << __FUNCTION__;

    auto total_start = std::chrono::steady_clock::now();

    word_cells = char_cells;
    auto after_copy = std::chrono::steady_clock::now();

    auto new_end
      = std::remove_if(word_cells.begin(),
		       word_cells.end(),
		       [](auto& cell) {
			 return utils::string::is_space(cell.text);
		       });
    word_cells.erase(new_end, word_cells.end());
    auto after_filter = std::chrono::steady_clock::now();

    double space_width_factor_for_merge_with_space = 2.0*space_width_factor_for_merge;

    cell_sanitizer.sanitize_bbox(word_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);
    auto after_sanitize = std::chrono::steady_clock::now();

    auto result = word_cells.get_compact();
    auto after_emit = std::chrono::steady_clock::now();

    sanitizer_profile::print("word_compact",
			     char_cells.size(),
			     word_cells.size(),
			     sanitizer_profile::elapsed_ms(total_start, after_copy),
			     sanitizer_profile::elapsed_ms(after_copy, after_filter),
			     sanitizer_profile::elapsed_ms(after_filter, after_sanitize),
			     sanitizer_profile::elapsed_ms(after_sanitize, after_emit),
			     sanitizer_profile::elapsed_ms(total_start, after_emit));

    return result;
  }

  nlohmann::json docling_sanitizer::create_line_cells(double horizontal_cell_tolerance,
						      bool enforce_same_font,
						      double space_width_factor_for_merge,
						      double space_width_factor_for_merge_with_space)
  {
    LOG_S(INFO) << __FUNCTION__ << " -> char_cells: " << char_cells.size();

    // do a deep copy
    line_cells = char_cells;

    LOG_S(INFO) << "initial line-cells: " << line_cells.size();
    
    cell_sanitizer.sanitize_bbox(line_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);
    
    LOG_S(INFO) << "initial line-cells: " << line_cells.size();

    return to_records("line");
  }  

  nlohmann::json docling_sanitizer::create_line_cells_table(double horizontal_cell_tolerance,
							    bool enforce_same_font,
							    double space_width_factor_for_merge,
							    double space_width_factor_for_merge_with_space)
  {
    LOG_S(INFO) << __FUNCTION__ << " -> char_cells: " << char_cells.size();

    auto total_start = std::chrono::steady_clock::now();

    line_cells = char_cells;
    auto after_copy = std::chrono::steady_clock::now();

    cell_sanitizer.sanitize_bbox(line_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);
    auto after_sanitize = std::chrono::steady_clock::now();

    auto result = line_cells.get();
    auto after_emit = std::chrono::steady_clock::now();

    sanitizer_profile::print("line_table",
			     char_cells.size(),
			     line_cells.size(),
			     sanitizer_profile::elapsed_ms(total_start, after_copy),
			     0.0,
			     sanitizer_profile::elapsed_ms(after_copy, after_sanitize),
			     sanitizer_profile::elapsed_ms(after_sanitize, after_emit),
			     sanitizer_profile::elapsed_ms(total_start, after_emit));

    return result;
  }  

  nlohmann::json docling_sanitizer::create_line_cells_table_compact(double horizontal_cell_tolerance,
								    bool enforce_same_font,
								    double space_width_factor_for_merge,
								    double space_width_factor_for_merge_with_space)
  {
    LOG_S(INFO) << __FUNCTION__ << " -> char_cells: " << char_cells.size();

    auto total_start = std::chrono::steady_clock::now();

    line_cells = char_cells;
    auto after_copy = std::chrono::steady_clock::now();

    cell_sanitizer.sanitize_bbox(line_cells,
				 horizontal_cell_tolerance,
				 enforce_same_font,
				 space_width_factor_for_merge,
				 space_width_factor_for_merge_with_space);
    auto after_sanitize = std::chrono::steady_clock::now();

    auto result = line_cells.get_compact();
    auto after_emit = std::chrono::steady_clock::now();

    sanitizer_profile::print("line_compact",
			     char_cells.size(),
			     line_cells.size(),
			     sanitizer_profile::elapsed_ms(total_start, after_copy),
			     0.0,
			     sanitizer_profile::elapsed_ms(after_copy, after_sanitize),
			     sanitizer_profile::elapsed_ms(after_sanitize, after_emit),
			     sanitizer_profile::elapsed_ms(total_start, after_emit));

    return result;
  }  

  
  
}

#endif
