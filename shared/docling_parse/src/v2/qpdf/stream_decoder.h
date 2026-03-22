//-*-C++-*-

#ifndef QPDF_STREAM_DECODER_H
#define QPDF_STREAM_DECODER_H

#include <exception>

namespace pdflib
{

  class qpdf_stream_decoder:
    public QPDFObjectHandle::ParserCallbacks
  {
  public:

    qpdf_stream_decoder(std::vector<qpdf_instruction>& stream_);
    ~qpdf_stream_decoder();

    void print();

    void decode(QPDFObjectHandle& content);

    // methods used by the QPDFObjectHandle::ParserCallbacks
    ///*virtual*/ void handleObject(QPDFObjectHandle obj) override;

    void handleObject(QPDFObjectHandle obj, size_t offset, size_t len) override;
    
    void contentSize(size_t len) override;

    void handleEOF() override;
    
  private:

    std::vector<qpdf_instruction>& stream;

    std::regex value_pattern_0;    
  };

  qpdf_stream_decoder::qpdf_stream_decoder(std::vector<qpdf_instruction>& stream_):
    QPDFObjectHandle::ParserCallbacks(),
    stream(stream_),

    value_pattern_0(R"(^(\d\.\d+)(\-\d+)$)")
  {}

  qpdf_stream_decoder::~qpdf_stream_decoder()
  {}

  void qpdf_stream_decoder::print()
  {
    for(auto row:stream)
      {
        LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
      }
  }

  void qpdf_stream_decoder::decode(QPDFObjectHandle& content)
  {
    LOG_S(INFO) << "start decoding content-stream: " << content.getTypeName() << " -> " << content.unparse();

    stream.clear();

    try
      {
        QPDFObjectHandle::parseContentStream(content, this);
      }
    catch(std::exception& e)
      {
        LOG_S(ERROR) << "QPDF encountered error (" << e.what() << ") during decoding";
      }

    LOG_S(WARNING) << "finished decoding content-stream!";
  }

  /*
  void qpdf_stream_decoder::handleObject(QPDFObjectHandle obj)
  {
    LOG_S(INFO) << __FUNCTION__;

    qpdf_instruction row;
    {
      row.key = obj.getTypeName();
      row.val = obj.unparse();
      row.obj = obj;

      //LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
    }

    stream.push_back(row);
  }
  */

  void qpdf_stream_decoder::handleObject(QPDFObjectHandle obj, size_t offset, size_t len)
  {
    qpdf_instruction row;
    row.obj = obj;

    if(obj.isNull())
      {
        // Reinterpret null as empty array (workaround for 'd' operator, Table 56)
        row.key = "parameter";
        row.val = "[]";
      }
    else if(obj.isOperator())
      {
        // Operators: need key + val for dispatch, but skip regex
        row.key = obj.getTypeName();
        row.val = obj.unparse();
      }
    else
      {
        row.key = obj.getTypeName();
        row.val = obj.unparse();

        // Only run the malformed-float regex on real/integer tokens that
        // contain a dash (extremely rare edge case: "1.234-5")
        if(obj.isNumber() && row.val.find('-') != std::string::npos)
          {
            std::smatch match;
            if(std::regex_match(row.val, match, value_pattern_0))
              {
                LOG_S(WARNING) << std::setw(12) << row.key << " | " << row.val
                               << " => new matched value: " << match[1];

                double value = std::stod(match[1].str());
                QPDFObjectHandle new_obj = QPDFObjectHandle::newReal(value);

                row.key = new_obj.getTypeName();
                row.val = new_obj.unparse();
                row.obj = new_obj;
              }
          }
      }

    stream.push_back(std::move(row));
  }

  void qpdf_stream_decoder::contentSize(size_t len)
  {
    //LOG_S(INFO) << __FUNCTION__ << ": " << len;
  }

  void qpdf_stream_decoder::handleEOF()
  {
    //LOG_S(INFO) << __FUNCTION__;
  }

}

#endif
